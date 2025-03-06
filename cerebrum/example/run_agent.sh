# run local agent
python cerebrum/run_agent.py \
    --mode local \
    --agent_path cerebrum/example/agents/test_agent \
    --task_input "What is the capital of United States?"

# run remote agent
python cerebrum/run_agent.py \
    --mode remote \
    --author demo_author \
    --name demo_agent \
    --task "What is the capital of United States?"
