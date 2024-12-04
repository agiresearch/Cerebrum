python cerebrum/example/aios_demo.py \
    --llm_name gemini-1.5-flash \
    --llm_backend google \
    --agent /common/home/km1558/projects/agiresearch/Cerebrum/cerebrum/example/agents/academic_agent \
    --task "Help me search AIOS and tell me what its core idea is."

python cerebrum/example/run_agent.py \
    --llm_name gemini-1.5-flash \
    --llm_backend google \
    --agent /common/home/km1558/projects/agiresearch/Cerebrum/cerebrum/example/agents/demo_agent \
    --task "Help me search AIOS and tell me what its core idea is."

python cerebrum/example/aios_demo.py \
    --llm_name gemini-1.5-flash \
    --llm_backend google \
    --agent example/math_agent \
    --task "Prove: 1! + 1/2! + 1/3! + 1/4! + ... = e - 1"

python cerebrum/example/run_agent.py \
    --llm_name gemini-1.5-flash \
    --llm_backend google \
    --agent /common/home/km1558/projects/agiresearch/Cerebrum/cerebrum/example/agents/demo_agent \
    --task "Give me a cocktail recipe with vodka, orange juice, and grenadine."

python cerebrum/example/run_agent.py \
    --llm_name gemini-1.5-flash \
    --llm_backend google \
    --agent /common/home/km1558/projects/agiresearch/Cerebrum/cerebrum/example/agents/story_teller \
    --task "Tell me a story about a dragon and a princess."

python cerebrum/example/aios_demo.py \
    --llm_name gpt-4o-mini \
    --llm_backend openai \
    --agent /common/home/km1558/projects/agiresearch/Cerebrum/cerebrum/example/agents/creation_agent \
    --task "Create a new animal that can fly and swim."

python cerebrum/example/aios_demo.py \
    --llm_name gpt-4o-mini \
    --llm_backend openai \
    --agent /common/home/km1558/projects/agiresearch/Cerebrum/cerebrum/example/agents/meme_creator \
    --task "Create a meme with a cat eating fish."

python cerebrum/example/aios_demo.py \
    --llm_name gpt-4o-mini \
    --llm_backend openai \
    --agent /common/home/km1558/projects/agiresearch/Cerebrum/cerebrum/example/agents/logo_creator \
    --task "Create a logo related to work."

python cerebrum/example/aios_demo.py \
    --llm_name gpt-4o-mini \
    --llm_backend openai \
    --agent /common/home/km1558/projects/agiresearch/Cerebrum/cerebrum/example/agents/logo_creator \
    --task "Create a logo related to an intelligent watch."

python cerebrum/example/aios_demo.py \
    --llm_name gpt-4o-mini \
    --llm_backend openai \
    --agent /common/home/km1558/projects/agiresearch/Cerebrum/cerebrum/example/agents/language_tutor \
    --task "Give me suggestions of how to learn French."

python cerebrum/example/aios_demo.py \
    --llm_name gpt-4o-mini \
    --llm_backend openai \
    --agent /common/home/km1558/projects/agiresearch/Cerebrum/cerebrum/example/agents/language_tutor \
    --task "Give me suggestions of how to use some greeting expressions in French."

python cerebrum/example/aios_demo.py \
    --llm_name meta-llama/Llama-3.2-1B-Instruct \
    --llm_backend huggingface \
    --agent example/tech_support_agent \
    --task "Help me solve the problem of my windows11 blue screen."

aios-basic-demo \
    --llm_name gpt-4o-mini \
    --llm_backend openai \
    --agent example/academic_agent \
    --task "Tell me the core idea of AIOS paper"

aios-basic-demo \
    --llm_name gpt-4o-mini \
    --llm_backend openai \
    --agent example/academic_agent \
    --task "Tell me the core idea of AIOS paper"

python cerebrum/example/aios_demo.py \
    --llm_name gemini-1.5-flash \
    --llm_backend google \
    --agent /common/home/km1558/projects/agiresearch/Cerebrum/cerebrum/example/agents/academic_agent \
    --task "Tell me the core idea of AIOS paper"

python cerebrum/example/aios_demo.py \
    --llm_name gpt-4o-mini \
    --llm_backend openai \
    --agent /common/home/km1558/projects/agiresearch/Cerebrum/cerebrum/example/agents/tech_support_agent \
    --task "Help me solve the problem of my windows11 blue screen."

run-agent \
    --llm_name gpt-4o-mini \
    --llm_backend openai \
    --agent demo_author/demo_agent \
    --task "Help me search AIOS and tell me what its core idea is."

python example/run_local_agent.py \
    --llm_name gemini-1.5-flash \
    --llm_backend google \
    --agent-path /common/home/km1558/projects/agiresearch/Cerebrum/example/agents/demo_agent \
    --task "Tell me the core idea of AIOS paper" \
    --aios-kernel-url http://localhost:8000

python example/run_hub_agent.py \
    --llm_name gemini-1.5-flash \
    --llm_backend google \
    --agent-name demo_author/demo_agent \
    --task "Tell me the core idea of AIOS paper" \
    --aios-kernel-url http://localhost:8000

python cerebrum/example/upload_agent_demo.py --agent_path /common/home/km1558/projects/agiresearch/Cerebrum/cerebrum/example/agents/demo_agent
# python cerebrum/example/upload_agent_demo.py --agent_path example/cocktail_mixlogist
python cerebrum/example/upload_tool_demo.py --tool_path /common/home/km1558/projects/agiresearch/Cerebrum/cerebrum/example/tools/arxiv