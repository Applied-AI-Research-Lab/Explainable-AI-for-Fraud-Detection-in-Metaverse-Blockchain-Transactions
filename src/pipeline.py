from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from time import perf_counter
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from .data_utils import LABEL_ORDER, decode_labels, load_dataset, prepare_train_test
from .explainability import ContributionExtractor
from .llm_engine import LLMDecisionEngine
from .modeling import build_models, compute_metrics, predict_models, train_models
from .reporting import (
    ensure_dirs,
    save_agreement_tables,
    save_class_distribution_table,
    save_confusion_figure,
    save_json,
    save_metric_barplot,
    save_metrics_tables,
)


def _build_llm_transaction_payload(raw_row: Dict, include_risk_score: bool) -> Dict:
    payload = dict(raw_row)
    # Never expose the ground-truth target to the LLM during evaluation/inference.
    payload.pop("anomaly", None)
    if not include_risk_score:
        payload.pop("risk_score", None)
    return payload


def _majority_vote(row: Dict[str, int]) -> int:
    votes = [row["logistic_regression"], row["random_forest"], row["xgboost"]]
    counts = Counter(votes)
    return counts.most_common(1)[0][0]


def _narrative_consistency(evidence_features: List[str], model_explanations: Dict) -> float:
    model_features = []
    for model_name in ["logistic_regression", "random_forest", "xgboost"]:
        model_features.extend([item["feature"] for item in model_explanations.get(model_name, [])[:5]])
    left = set(evidence_features)
    right = set(model_features)
    if not left or not right:
        return 0.0
    return len(left.intersection(right)) / len(left.union(right))


def run(args: argparse.Namespace) -> None:
    np.random.seed(args.random_state)
    print("[1/8] Initializing output directories...", flush=True)
    paths = ensure_dirs(args.output_dir)

    print("[2/8] Loading dataset...", flush=True)
    df = load_dataset(args.data_path, max_rows=args.max_rows)
    print(f"Loaded rows: {len(df)}", flush=True)

    print("[3/8] Preparing train/test features...", flush=True)
    data = prepare_train_test(
        df=df,
        test_size=args.test_size,
        random_state=args.random_state,
        include_risk_score=args.include_risk_score,
    )
    print(
        f"Prepared shapes -> X_train: {data.X_train.shape}, X_test: {data.X_test.shape}",
        flush=True,
    )

    print("[4/8] Building and training classical models...", flush=True)
    models = build_models(random_state=args.random_state, n_classes=len(LABEL_ORDER))
    models = train_models(models, data.X_train, data.y_train)

    for name, model in models.items():
        joblib.dump(model, os.path.join(paths["models"], f"{name}.joblib"))

    print("[5/8] Running classical model predictions and metrics...", flush=True)
    preds = predict_models(models, data.X_test)

    # Measure per-row inference latency for each classical model.
    model_pred_times: Dict[str, np.ndarray] = {}
    for model_name, model in models.items():
        row_times = np.zeros(data.X_test.shape[0], dtype=float)
        for i in range(data.X_test.shape[0]):
            t0 = perf_counter()
            model.predict_proba(data.X_test[i : i + 1])
            row_times[i] = perf_counter() - t0
        model_pred_times[model_name] = row_times

    metrics_rows = []
    reports = {}
    pred_df = pd.DataFrame(
        {
            "y_true": data.y_test,
            "y_true_label": decode_labels(data.y_test),
        }
    )

    for model_name, output in preds.items():
        metric_dict, report_dict = compute_metrics(
            y_true=data.y_test,
            y_pred=output["y_pred"],
            y_prob=output["y_prob"],
            n_classes=len(LABEL_ORDER),
        )
        metric_dict["model"] = model_name
        metrics_rows.append(metric_dict)
        reports[model_name] = report_dict

        pred_df[f"pred_{model_name}"] = output["y_pred"]
        pred_df[f"pred_label_{model_name}"] = decode_labels(output["y_pred"])

        for i, label in enumerate(LABEL_ORDER):
            pred_df[f"prob_{model_name}_{label}"] = output["y_prob"][:, i]

        pred_df[f"time_{model_name}_sec"] = model_pred_times[model_name]

        save_confusion_figure(paths, model_name, data.y_test, output["y_pred"])

    metric_df = save_metrics_tables(paths, metrics_rows)
    save_metric_barplot(paths, metric_df)
    save_json(os.path.join(paths["metrics"], "classification_reports.json"), reports)
    pred_df.to_csv(os.path.join(paths["predictions"], "test_predictions.csv"), index=False)
    save_class_distribution_table(paths, decode_labels(data.y_test))

    print("[6/8] Initializing LLM aggregation stage...", flush=True)
    llm_engine = LLMDecisionEngine(
        mode=args.llm_mode,
        model_name_or_path=args.llm_model,
        max_new_tokens=args.llm_max_new_tokens,
        max_retries=args.llm_max_retries,
        backend=args.llm_backend,
        load_in_4bit=args.llm_load_in_4bit,
        max_seq_length=args.llm_max_seq_length,
        dtype_name=args.llm_dtype,
    )

    max_samples = min(args.llm_max_samples, data.X_test.shape[0])
    sampled_idx = np.random.choice(data.X_test.shape[0], size=max_samples, replace=False)

    background_size = min(args.shap_background_size, data.X_train.shape[0])
    bg_idx = np.random.choice(data.X_train.shape[0], size=background_size, replace=False)
    extractor = ContributionExtractor(
        models=models,
        feature_names=data.feature_names,
        X_background=data.X_train[bg_idx],
    )

    llm_rows = []
    llm_true = []
    llm_pred = []
    majority_pred = []
    consistency_scores = []
    llm_inference_times = []

    print(f"[7/8] LLM aggregation over {max_samples} sampled rows...", flush=True)
    for idx in tqdm(sampled_idx, desc="LLM aggregation"):
        row_features = data.X_test[idx]
        row_raw = data.test_raw.iloc[idx].to_dict()
        llm_transaction = _build_llm_transaction_payload(
            raw_row=row_raw,
            include_risk_score=args.include_risk_score,
        )

        model_outputs = {}
        model_row_votes = {}
        for model_name, output in preds.items():
            pred_cls = int(output["y_pred"][idx])
            probs = output["y_prob"][idx].tolist()
            model_outputs[model_name] = {
                "predicted_class": LABEL_ORDER[pred_cls],
                "probabilities": probs,
            }
            model_row_votes[model_name] = pred_cls

        explanation_seed_class = int(preds["xgboost"]["y_pred"][idx])
        model_explanations = extractor.explain_row(row_features, predicted_class_idx=explanation_seed_class, top_k=5)

        llm_t0 = perf_counter()
        llm_decision = llm_engine.decide(
            transaction=llm_transaction,
            model_outputs=model_outputs,
            model_explanations=model_explanations,
        )
        llm_time = perf_counter() - llm_t0

        y_true_int = int(data.y_test[idx])
        y_pred_int = LABEL_ORDER.index(llm_decision.final_label)
        majority_int = _majority_vote(model_row_votes)

        consistency = _narrative_consistency(
            evidence_features=llm_decision.evidence_features,
            model_explanations=model_explanations,
        )

        llm_true.append(y_true_int)
        llm_pred.append(y_pred_int)
        majority_pred.append(majority_int)
        consistency_scores.append(consistency)
        llm_inference_times.append(llm_time)

        llm_rows.append(
            {
                "index": int(idx),
                "y_true": y_true_int,
                "y_true_label": LABEL_ORDER[y_true_int],
                "llm_pred": y_pred_int,
                "llm_pred_label": llm_decision.final_label,
                "risk_score": llm_decision.risk_score,
                "llm_confidence": llm_decision.confidence,
                "majority_vote_label": LABEL_ORDER[majority_int],
                "narrative_consistency": consistency,
                "evidence_features": json.dumps(llm_decision.evidence_features),
                "llm_explanation": llm_decision.explanation,
                "llm_inference_time_sec": llm_time,
                "raw_response": llm_decision.raw_response,
                "model_explanations": json.dumps(model_explanations),
            }
        )

    print("[8/8] Writing reports and final artifacts...", flush=True)
    llm_true_arr = np.array(llm_true)
    llm_pred_arr = np.array(llm_pred)

    llm_probs_proxy = np.zeros((len(llm_pred_arr), len(LABEL_ORDER)))
    for i, cls in enumerate(llm_pred_arr):
        llm_probs_proxy[i, cls] = 1.0

    llm_metrics, llm_report = compute_metrics(
        y_true=llm_true_arr,
        y_pred=llm_pred_arr,
        y_prob=llm_probs_proxy,
        n_classes=len(LABEL_ORDER),
    )
    llm_metrics["model"] = f"llm_{args.llm_mode}"

    metrics_rows_with_llm = metrics_rows + [llm_metrics]
    metric_df = save_metrics_tables(paths, metrics_rows_with_llm)
    save_metric_barplot(paths, metric_df)
    reports[f"llm_{args.llm_mode}"] = llm_report
    save_json(os.path.join(paths["metrics"], "classification_reports.json"), reports)

    llm_df = pd.DataFrame(llm_rows)
    llm_df.to_csv(os.path.join(paths["predictions"], "llm_predictions.csv"), index=False)
    save_confusion_figure(paths, f"llm_{args.llm_mode}", llm_true_arr, llm_pred_arr)

    # Save timing summary for classical and LLM stages.
    timing_rows = []
    for model_name, times_arr in model_pred_times.items():
        timing_rows.append(
            {
                "model": model_name,
                "n_predictions": int(len(times_arr)),
                "total_time_sec": float(np.sum(times_arr)),
                "mean_time_sec": float(np.mean(times_arr)),
                "median_time_sec": float(np.median(times_arr)),
                "p95_time_sec": float(np.percentile(times_arr, 95)),
            }
        )

    llm_times_arr = np.array(llm_inference_times, dtype=float)
    if llm_times_arr.size > 0:
        timing_rows.append(
            {
                "model": f"llm_{args.llm_mode}",
                "n_predictions": int(len(llm_times_arr)),
                "total_time_sec": float(np.sum(llm_times_arr)),
                "mean_time_sec": float(np.mean(llm_times_arr)),
                "median_time_sec": float(np.median(llm_times_arr)),
                "p95_time_sec": float(np.percentile(llm_times_arr, 95)),
            }
        )

    pd.DataFrame(timing_rows).to_csv(
        os.path.join(paths["metrics"], "prediction_time_summary.csv"),
        index=False,
    )

    majority_arr = np.array(majority_pred)
    majority_wrong = majority_arr != llm_true_arr
    llm_correct = llm_pred_arr == llm_true_arr
    correction_den = int(np.sum(majority_wrong))
    correction_num = int(np.sum(majority_wrong & llm_correct))

    unanimity_classical = float(
        np.mean(
            (preds["logistic_regression"]["y_pred"][sampled_idx] == preds["random_forest"]["y_pred"][sampled_idx])
            & (preds["random_forest"]["y_pred"][sampled_idx] == preds["xgboost"]["y_pred"][sampled_idx])
        )
    )
    unanimity_all = float(
        np.mean(
            (preds["logistic_regression"]["y_pred"][sampled_idx] == preds["random_forest"]["y_pred"][sampled_idx])
            & (preds["random_forest"]["y_pred"][sampled_idx] == preds["xgboost"]["y_pred"][sampled_idx])
            & (preds["xgboost"]["y_pred"][sampled_idx] == llm_pred_arr)
        )
    )

    agreement_stats = {
        "llm_samples": int(len(sampled_idx)),
        "unanimity_rate_classical": unanimity_classical,
        "unanimity_rate_classical_plus_llm": unanimity_all,
        "llm_correction_rate_vs_majority": float(correction_num / correction_den) if correction_den > 0 else 0.0,
        "mean_narrative_consistency": float(np.mean(consistency_scores)) if consistency_scores else 0.0,
    }
    save_agreement_tables(paths, agreement_stats)

    summary = {
        "data_rows": int(df.shape[0]),
        "train_rows": int(data.X_train.shape[0]),
        "test_rows": int(data.X_test.shape[0]),
        "class_order": LABEL_ORDER,
        "llm_mode": args.llm_mode,
        "llm_samples": int(len(sampled_idx)),
        "agreement_stats": agreement_stats,
    }
    save_json(os.path.join(paths["metrics"], "run_summary.json"), summary)

    print("Pipeline completed successfully.")
    print(f"Outputs written to: {args.output_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Metaverse transaction explainable risk pipeline")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--include-risk-score", action="store_true")

    parser.add_argument("--llm-mode", type=str, choices=["heuristic", "hf_local"], default="heuristic")
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--llm-backend", choices=["transformers", "unsloth"], default="transformers")
    parser.add_argument("--llm-load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--llm-max-seq-length", type=int, default=2048)
    parser.add_argument("--llm-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--llm-max-samples", type=int, default=1000)
    parser.add_argument("--llm-max-new-tokens", type=int, default=220)
    parser.add_argument("--llm-max-retries", type=int, default=3)

    parser.add_argument("--shap-background-size", type=int, default=1000)
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    run(parser.parse_args())
