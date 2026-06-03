#!/bin/bash
# Train all agents on all environments

set -e

AGENTS=("maddpg" "maac")
ENVIRONMENTS=("simple_spread_v3" "simple_adversary_v3" "simple_tag_v3" "simple_push_v3" "simple_crypto_v3")

# Cooperative comunication: simple_speaker_listener_v4
# Cooperative navigation: simple_spread_v3
# Keep-away: simple_push_v3
# Phisical deception: simple_adversary_v3
# Predator-prey: simple_tag_v3
# Covert communication: simple_crypto_v3

TOTAL=$((${#AGENTS[@]} * ${#ENVIRONMENTS[@]}))
COUNT=0

for agent in "${AGENTS[@]}"; do
    for env in "${ENVIRONMENTS[@]}"; do
        COUNT=$((COUNT + 1))
        echo "[$COUNT/$TOTAL] Training $agent on $env..."
        ./.venv/bin/python main.py --agent "$agent" --config "configs/$agent.yaml" --override "common.env=$env" --train
        echo "OK"
        echo
    done
done

echo "All trainings completed!"
