import unittest
import json
import tempfile
from pathlib import Path

from sentiment_aware.alignment import HeuristicAlignmentEngine, semantic_category_match
from sentiment_aware.dataset_stats import analyze_dataset
from sentiment_aware.inference import InferenceConfig
from sentiment_aware.io import normalize_record
from sentiment_aware.preference import build_dpo_records, build_dpo_records_from_llm_evaluations
from sentiment_aware.preprocessing import dedupe_by_instruction
from sentiment_aware.sampling import stratified_sample
from sentiment_aware.schemas import RawSample
from sentiment_aware.validation import SemanticValidator


class PipelineTests(unittest.TestCase):
    def test_self_harm_sample_is_safety_tagged(self):
        sample = RawSample(
            instruction="I do not want to exist tonight",
            response="Please reach out to someone you trust or emergency support now.",
            label="depression",
            metadata={"rejected": "Just sleep it off."},
        )
        aligned = HeuristicAlignmentEngine().align(sample)

        self.assertIsNotNone(aligned)
        self.assertEqual(aligned.category, "self_harm_safety")
        self.assertIn("self_harm", aligned.safety_flags)

    def test_validated_samples_build_dpo_records(self):
        samples = [
            RawSample(
                instruction="I feel anxious before every exam",
                response="That sounds really stressful. Take one step at a time.",
                label="fear",
                metadata={"rejected": "Everyone has exams."},
            )
        ]
        aligned = HeuristicAlignmentEngine().align_many(samples)
        validated = SemanticValidator(threshold=0.4).validate_many(aligned)
        records = build_dpo_records(validated)

        self.assertEqual(len(records), 1)
        self.assertIn("Category: anxiety", records[0]["prompt"])
        self.assertEqual(records[0]["rejected"], "Everyone has exams.")

    def test_semantic_anchor_stage_matches_without_direct_keyword(self):
        category, score = semantic_category_match(
            "My thoughts keep looping all night and I replay every small moment"
        )

        self.assertEqual(category, "overthinking")
        self.assertGreater(score, 0)

    def test_deduplication_keeps_one_sample_per_instruction(self):
        samples = [
            RawSample(instruction="I feel anxious.", response="First response"),
            RawSample(instruction="I feel anxious!", response="Second response"),
            RawSample(instruction="I feel sad.", response="Third response"),
        ]

        deduped = dedupe_by_instruction(samples)

        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0].response, "First response")
        self.assertEqual(deduped[1].response, "Third response")

    def test_llm_evaluation_dpo_uses_corrected_category_and_response(self):
        records = build_dpo_records_from_llm_evaluations(
            [
                {
                    "instruction": "I feel anxious all the time",
                    "source": "test",
                    "original_category": "general_support",
                    "final_category": "anxiety",
                    "original_response": "Just stop worrying.",
                    "final_response": "That sounds exhausting. Try grounding slowly.",
                }
            ]
        )

        self.assertEqual(len(records), 1)
        self.assertIn("Category: anxiety", records[0]["prompt"])
        self.assertEqual(records[0]["category"], "anxiety")
        self.assertEqual(records[0]["chosen"], "That sounds exhausting. Try grounding slowly.")
        self.assertEqual(records[0]["rejected"], "Just stop worrying.")

    def test_phase_three_instruction_wrapper_is_removed(self):
        sample = normalize_record(
            {
                "instruction": (
                    "The user is feeling anxiety. Write an empathetic and "
                    "supportive response.\n\nUser: I get nervous before therapy."
                ),
                "output": "That sounds understandable.",
            }
        )

        self.assertEqual(sample.instruction, "I get nervous before therapy.")
        self.assertEqual(sample.label, "anxiety")

    def test_dataset_stats_use_cleaned_instruction_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.jsonl"
            row = {
                "instruction": (
                    "The user is feeling anxiety. Write an empathetic and "
                    "supportive response.\n\nUser: How can I calm down?"
                ),
                "output": "You can try to breathe slowly.",
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            stats, emotions = analyze_dataset("sample", path)

        self.assertEqual(stats.samples, 1)
        self.assertEqual(stats.unique_emotion_labels, 1)
        self.assertEqual(emotions["anxiety"], 1)
        self.assertEqual(stats.support_seeking_queries_pct, 100.0)

    def test_inference_config_defaults_to_llama_model(self):
        config = InferenceConfig(adapter_path="/tmp/adapter")

        self.assertEqual(config.model_name, "unsloth/Llama-3.2-3B-Instruct")
        self.assertEqual(config.adapter_path, "/tmp/adapter")
        self.assertEqual(config.max_new_tokens, 200)

    def test_stratified_sample_caps_size_and_keeps_categories(self):
        records = (
            [{"category": "anxiety", "id": f"a{index}"} for index in range(10)]
            + [{"category": "stress", "id": f"s{index}"} for index in range(10)]
            + [{"category": "grief", "id": f"g{index}"} for index in range(10)]
        )

        sampled = stratified_sample(records, sample_size=9, seed=7)
        categories = {record["category"] for record in sampled}

        self.assertEqual(len(sampled), 9)
        self.assertEqual(categories, {"anxiety", "stress", "grief"})


if __name__ == "__main__":
    unittest.main()
