## Setup

1. Configure `.env`:
   ```bash
   OPENROUTER_API_KEY=your_api_key_here
   ```
2. Place `mimic_iii.db` and `valid_preprocessed.jsonl` under `data/ehrsql-ehragent/mimic_iii/`.

## Quick Start

```bash
# Run default evaluation (deepseek-v4-flash-0731, 50 questions)
./run_eval.sh

# Run specific model
./run_eval.sh gpt-oss-120b-openrouter 50 0
```
