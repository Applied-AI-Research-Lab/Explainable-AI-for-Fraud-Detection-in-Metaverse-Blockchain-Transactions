from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .data_utils import LABEL_ORDER


@dataclass
class LLMDecision:
    final_label: str
    risk_score: float
    explanation: str
    confidence: float
    evidence_features: List[str]
    raw_response: str


class LLMDecisionEngine:
    def __init__(
        self,
        mode: str = "heuristic",
        model_name_or_path: str | None = None,
        max_new_tokens: int = 220,
        max_retries: int = 3,
        backend: str = "transformers",
        load_in_4bit: bool = True,
        max_seq_length: int = 2048,
        dtype_name: str = "auto",
    ):
        self.mode = mode
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.max_retries = max(1, int(max_retries))
        self.backend = backend
        self.load_in_4bit = load_in_4bit
        self.max_seq_length = max_seq_length
        self.dtype_name = dtype_name
        self._generator = None
        self._model = None
        self._tokenizer = None

    def _risk_score_from_prob(self, prob_vec: np.ndarray) -> float:
        anchor = np.array([15.0, 55.0, 90.0], dtype=float)
        return float(np.dot(prob_vec, anchor))

    def _label_from_risk(self, risk_score: float) -> str:
        if risk_score >= 70:
            return "high_risk"
        if risk_score >= 35:
            return "moderate_risk"
        return "low_risk"

    def _build_prompt(self, transaction: Dict, model_outputs: Dict, model_explanations: Dict) -> str:
        return (
            "You are a senior fraud analyst for metaverse blockchain transactions. "
            "Synthesize model evidence into a business-actionable verdict. Return strict JSON only with keys: "
            "final_label, risk_score, confidence, evidence_features, explanation.\n\n"
            f"Transaction: {json.dumps(transaction, default=str)}\n"
            f"Model Outputs: {json.dumps(model_outputs)}\n"
            f"Feature Explanations: {json.dumps(model_explanations)}\n\n"
            "Rules: risk_score must be 0-100. final_label must be one of low_risk, moderate_risk, high_risk. "
            "explanation should be concise and mention business impact and recommended action."
        )

    def _init_generator(self):
        if self._generator is not None:
            return
        if self.mode != "hf_local":
            return
        if not self.model_name_or_path:
            raise ValueError("llm_model must be provided when mode=hf_local")
        if self.backend == "unsloth":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("Unsloth backend requires CUDA GPU.")

            try:
                from unsloth import FastLanguageModel
                from unsloth.chat_templates import get_chat_template
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "Unsloth is not installed in the active environment."
                ) from exc

            dtype_map = {
                "auto": None,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            if self.dtype_name not in dtype_map:
                raise ValueError(f"Unsupported dtype: {self.dtype_name}")

            self._model, self._tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name_or_path,
                max_seq_length=self.max_seq_length,
                dtype=dtype_map[self.dtype_name],
                load_in_4bit=self.load_in_4bit,
            )
            self._tokenizer = get_chat_template(self._tokenizer, chat_template="llama-3.1")
            FastLanguageModel.for_inference(self._model)
            self._generator = "unsloth"
            return

        from transformers import pipeline

        self._generator = pipeline(
            task="text-generation",
            model=self.model_name_or_path,
            device_map="auto",
        )

    def _heuristic(self, model_outputs: Dict, model_explanations: Dict) -> LLMDecision:
        weights = {
            "logistic_regression": 0.20,
            "random_forest": 0.35,
            "xgboost": 0.45,
        }
        stacked = []
        for name, output in model_outputs.items():
            stacked.append(weights[name] * np.array(output["probabilities"], dtype=float))
        prob_vec = np.sum(stacked, axis=0)
        prob_vec = prob_vec / np.sum(prob_vec)

        risk_score = self._risk_score_from_prob(prob_vec)
        final_label = self._label_from_risk(risk_score)
        confidence = float(np.max(prob_vec))

        top_features = []
        for model_name in ["random_forest", "xgboost", "logistic_regression"]:
            feats = model_explanations.get(model_name, [])
            top_features.extend([item["feature"] for item in feats[:2]])
        evidence_features = list(dict.fromkeys(top_features))[:6]

        explanation = (
            f"Aggregated model confidence indicates {final_label}. "
            f"Main evidence features: {', '.join(evidence_features)}. "
            "Recommended business action: apply stepped verification and transaction monitoring before settlement."
        )

        return LLMDecision(
            final_label=final_label,
            risk_score=risk_score,
            explanation=explanation,
            confidence=confidence,
            evidence_features=evidence_features,
            raw_response="heuristic",
        )

    def _parse_json(self, text: str) -> Dict:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return {}

    def decide(self, transaction: Dict, model_outputs: Dict, model_explanations: Dict) -> LLMDecision:
        if self.mode == "heuristic":
            return self._heuristic(model_outputs=model_outputs, model_explanations=model_explanations)

        self._init_generator()
        prompt = self._build_prompt(transaction, model_outputs, model_explanations)
        last_response = ""
        parsed: Dict = {}
        for _attempt in range(1, self.max_retries + 1):
            try:
                if self.backend == "unsloth":
                    import torch

                    messages = [{"role": "user", "content": prompt}]
                    input_ids = self._tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    ).to("cuda")
                    output = self._model.generate(
                        input_ids=input_ids,
                        max_new_tokens=self.max_new_tokens,
                        use_cache=True,
                        do_sample=False,
                    )
                    generated_tokens = output[0, input_ids.shape[-1] :]
                    generated = self._tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                else:
                    generated = self._generator(
                        prompt,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        return_full_text=False,
                    )[0]["generated_text"]
                last_response = generated
                parsed = self._parse_json(generated)
                if parsed:
                    break
            except Exception as exc:
                last_response = f"GENERATION_EXCEPTION: {exc}"

        if not parsed:
            raise RuntimeError(
                "LLM did not produce valid JSON within max retries. "
                f"retries={self.max_retries}\n\n"
                f"PROMPT:\n{prompt}\n\n"
                f"LAST_RESPONSE:\n{last_response}"
            )

        final_label = parsed.get("final_label", "moderate_risk")
        if final_label not in LABEL_ORDER:
            final_label = "moderate_risk"

        risk_score = float(parsed.get("risk_score", 55.0))
        risk_score = max(0.0, min(100.0, risk_score))

        confidence = float(parsed.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        evidence_features = parsed.get("evidence_features", [])
        if not isinstance(evidence_features, list):
            evidence_features = []
        evidence_features = [str(v) for v in evidence_features][:8]

        explanation = str(parsed.get("explanation", "No explanation provided by model."))

        return LLMDecision(
            final_label=final_label,
            risk_score=risk_score,
            explanation=explanation,
            confidence=confidence,
            evidence_features=evidence_features,
            raw_response=last_response,
        )
