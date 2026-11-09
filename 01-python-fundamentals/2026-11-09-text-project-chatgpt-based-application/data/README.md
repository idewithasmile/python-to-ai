# Natural Language Processing & Large Language Model Benchmark Datasets

This directory provides curated benchmark datasets and evaluation suites designed for probing, validating, and testing Large Language Model (LLM) prompt engineering patterns and conversational capabilities.

---

## Dataset Directory Structure

```text
data/
├── README.md                                # Dataset documentation and schema specifications
├── sentiment_classification_benchmarks.json # Binary and 3-way sentiment classification evaluation suite
├── multilabel_aspect_samples.json           # Aspect-based multi-label restaurant review corpus
├── parallel_translation_corpus.json         # Parallel Vietnamese-English machine translation evaluation pairs
├── reading_comprehension_qa.json            # Machine reading comprehension passages and QA benchmarks
├── reasoning_cot_benchmarks.json            # Arithmetic, logic, and parity reasoning benchmarks (Direct vs CoT)
└── code_debugging_testcases.json            # Code synthesis, refactoring, and automated bug detection cases
```

---

## 1. Sentiment Classification Benchmarks (`sentiment_classification_benchmarks.json`)

### Description
Contains 10 curated English sentences exhibiting positive, negative, and nuanced neutral sentiment. Used to evaluate zero-shot, one-shot, and few-shot classification prompts, as well as chain-of-thought rationale generation.

### Schema
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "SentimentBenchmarkRecord",
  "type": "object",
  "required": ["id", "text", "label", "explanation"],
  "properties": {
    "id": { "type": "string", "description": "Unique identifier (e.g. sent-001)" },
    "text": { "type": "string", "description": "Input sentence or customer review" },
    "label": { "type": "string", "enum": ["positive", "neutral", "negative"] },
    "explanation": { "type": "string", "description": "Linguistic rationale explaining why the label was assigned" }
  }
}
```

---

## 2. Multi-Label Aspect Review Corpus (`multilabel_aspect_samples.json`)

### Description
Contains dining, hospitality, and product reviews labeled across five fine-grained aspect categories: `service`, `food`, `price`, `ambience`, and `miscellaneous`. Used to test multi-label classification prompts and structured JSON extraction.

### Schema
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "MultilabelAspectRecord",
  "type": "object",
  "required": ["id", "text", "labels", "aspect_breakdown"],
  "properties": {
    "id": { "type": "string", "description": "Unique identifier (e.g. aspect-001)" },
    "text": { "type": "string", "description": "Customer review spanning single or multiple aspects" },
    "labels": {
      "type": "array",
      "items": { "type": "string", "enum": ["service", "food", "price", "ambience", "miscellaneous"] }
    },
    "aspect_breakdown": {
      "type": "object",
      "additionalProperties": { "type": "string" },
      "description": "Mapping of detected aspect label to the corresponding textual phrase"
    }
  }
}
```

---

## 3. Parallel Translation Corpus (`parallel_translation_corpus.json`)

### Description
Bilingual parallel text pairs (Vietnamese to English) across academic, vocabulary, conversational, and technical domains. Used to evaluate translation templates TP1 (`Translate from [SRC] to [TGT]`), TP2 (`What do these sentences mean in [TGT]?`), and TP3 (`Please provide the [TGT] translation`).

### Schema
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "ParallelTranslationRecord",
  "type": "object",
  "required": ["id", "source_lang", "target_lang", "source_text", "reference_translation", "domain"],
  "properties": {
    "id": { "type": "string" },
    "source_lang": { "type": "string", "pattern": "^[a-z]{2}$" },
    "target_lang": { "type": "string", "pattern": "^[a-z]{2}$" },
    "source_text": { "type": "string" },
    "reference_translation": { "type": "string" },
    "domain": { "type": "string", "enum": ["academic", "vocabulary", "conversational", "technical", "daily-life"] }
  }
}
```

---

## 4. Reading Comprehension & QA Suite (`reading_comprehension_qa.json`)

### Description
Context-grounded reading passages paired with factual questions, multiple-choice options, and unanswerable challenge queries. Used to benchmark closed-domain extractive question answering, abstractive summarization, and hallucination avoidance / abstention.

### Schema
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "QuestionAnsweringRecord",
  "type": "object",
  "required": ["id", "type", "context", "question", "target_answer"],
  "properties": {
    "id": { "type": "string" },
    "type": { "type": "string", "enum": ["closed_domain_biography", "closed_domain_literature", "multiple_choice_science", "closed_domain_sports", "unanswerable_challenge"] },
    "context": { "type": "string", "description": "Passage text providing source grounding" },
    "question": { "type": "string" },
    "options": { "type": "object", "description": "Optional mapping of option keys A-D to choice strings" },
    "target_answer": { "type": "string" },
    "extractive_span": { "type": "string", "description": "Exact textual substring from context containing the answer" },
    "abstention_instruction": { "type": "string", "description": "Expected refusal or abstention directive when context is insufficient" }
  }
}
```

---

## 5. Reasoning & Chain-of-Thought Benchmarks (`reasoning_cot_benchmarks.json`)

### Description
Challenging symbolic, mathematical, and algorithmic reasoning problems where naive zero-shot LLM prompts often fail due to token prediction heuristics, contrasted with multi-step Chain-of-Thought (CoT) reasoning trajectories.

### Schema
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "ReasoningBenchmarkRecord",
  "type": "object",
  "required": ["id", "type", "prompt", "cot_steps", "expected_result"],
  "properties": {
    "id": { "type": "string" },
    "type": { "type": "string", "enum": ["arithmetic_multiplication", "odd_sum_parity", "letter_counting_tokenization_trap"] },
    "prompt": { "type": "string" },
    "direct_answer": { "type": "string" },
    "direct_failure_mode": { "type": "string" },
    "cot_steps": { "type": "array", "items": { "type": "string" } },
    "expected_result": { "type": ["string", "number", "integer"] }
  }
}
```

---

## 6. Code Debugging & Synthesis Cases (`code_debugging_testcases.json`)

### Description
Python code generation and bug remediation cases including syntax errors, logic/operator errors, and algorithmic boundary edge cases. Paired with unit test cases and reference corrected code.

### Schema
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "CodeDebuggingRecord",
  "type": "object",
  "required": ["id", "name", "description", "buggy_code", "fixed_code", "test_inputs", "expected_output"],
  "properties": {
    "id": { "type": "string" },
    "name": { "type": "string" },
    "description": { "type": "string" },
    "buggy_code": { "type": "string" },
    "fixed_code": { "type": "string" },
    "test_inputs": { "type": "array" },
    "expected_output": { "type": ["number", "boolean", "string", "array"] }
  }
}
```
