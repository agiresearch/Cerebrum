from cerebrum.manager.agent import AgentManager

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload agents")
    parser.add_argument(
        "--agent_path",
        required=True
    )
    args = parser.parse_args()

    manager = AgentManager('https://app.aios.foundation')

    agent_package = manager.package_agent(args.agent_path)

    manager.upload_agent(agent_package)
