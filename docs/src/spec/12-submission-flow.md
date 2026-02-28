# Submission Flow

```bash
# 1. Generate HuggingFace model card
apr eval final.apr --generate-card

# 2. Export to HuggingFace-compatible format
apr export final.apr --format safetensors -o submission/

# 3. Publish to HuggingFace Hub
apr publish submission/ --repo paiml/qwen-coder-7b-apr --private

# 4. Submit to leaderboard (via HF evaluation queue)
# The leaderboard pulls from your HF repo and runs evaluation
```
