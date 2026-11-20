#!/bin/sh
# Train all agents on all environments

set -e

AGENTS="mat happo"
ENVIRONMENTS="simple_speaker_listener_v4 simple_spread_v3 simple_adversary_v3 simple_tag_v3 simple_push_v3 simple_crypto_v3"

# Cooperative comunication: simple_speaker_listener_v4
# Cooperative navigation: simple_spread_v3
# Keep-away: simple_push_v3
# Phisical deception: simple_adversary_v3
# Predator-prey: simple_tag_v3
# Covert communication: simple_crypto_v3

AGENT_COUNT=$(set -- $AGENTS; echo $#)
ENVIRONMENT_COUNT=$(set -- $ENVIRONMENTS; echo $#)
TOTAL=$((AGENT_COUNT * ENVIRONMENT_COUNT))
COUNT=0

for agent in $AGENTS; do
    for env in $ENVIRONMENTS; do
        COUNT=$((COUNT + 1))
        echo "[$COUNT/$TOTAL] Training $agent on $env..."
        ./.venv/bin/python main.py --agent "$agent" --config "configs/$agent.yaml" --override "common.env=$env" --override "common.episodes=25000" --train
        echo "OK"
        echo
    done
done

echo "All trainings completed!"

# Cross-algorithm comparison. Each run already wrote its own figures via the
# auto_report hook; this is the four-way overlay, which can only be produced
# once every algorithm has finished. --common_horizon truncates all curves to
# the shortest run so unequal training budgets do not skew the comparison.
echo
echo "Generating comparison report..."
./.venv/bin/python reports/report.py --env_type mpe2 --common_horizon
echo "Report done."
