from cerebrum.interface import AutoLLM, AutoTool

tool_name = "/common/home/km1558/projects/agiresearch/Cerebrum/example/tools/arxiv"

tool = AutoTool.from_preloaded(tool_name)

print(tool)