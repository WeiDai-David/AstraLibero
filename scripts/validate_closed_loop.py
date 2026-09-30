"""Real simulator with a labelled HTTP fixture; never an Astra success claim."""
import argparse
import json
import os
from pathlib import Path
import httpx
import imageio.v2 as imageio
import numpy as np
from src.agents.astra_agent import AstraAgent
from src.controllers.semantic_controller import ModeBasedSemanticController
from src.envs.libero_env import LiberoEnvWrapper
from src.rollout import run_rollout
from src.schemas.action import SemanticAction
from src.utils.config import load_config


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/libero_spatial.yaml")
    args=parser.parse_args()
    cfg=load_config(args.config)
    # Process-local fake credentials; MockTransport never makes a network request.
    os.environ.update(ASTRA_API_KEY="diagnostic-fixture",ASTRA_BASE_URL="https://diagnostic.invalid/v1",ASTRA_MODEL="DIAGNOSTIC_HTTP_FIXTURE_NOT_ASTRA")
    seen=[]
    def respond(request):
        payload=json.loads(request.content)
        content=payload["messages"][1]["content"]
        current=json.loads(content[0]["text"])
        seen.append((current["observation_step"],content[2]["image_url"]["url"]))
        action=SemanticAction.noop().model_dump(mode="json")
        # A bounded direction probe driven by current observation sequence.
        action["x"]="POSITIVE" if current["observation_step"]==0 else "ZERO"
        return httpx.Response(200,json={"choices":[{"message":{"content":json.dumps(action)}}]})
    agent=AstraAgent(client=httpx.Client(transport=httpx.MockTransport(respond)))
    env=LiberoEnvWrapper(cfg["environment"])
    try:
        env.reset(cfg["environment"]["task_id"],cfg["environment"]["seed"])
        controller=ModeBasedSemanticController(cfg["controller"],env.action_spec)
        result=run_rollout(env,agent,controller,max_steps=3,history_length=2,
            output_root=cfg["rollout"]["output_root"],category="integration_test",
            metadata={"diagnostic":True,"provider":"local HTTP fixture; no external model call"})
        assert [item[0] for item in seen]==[0,1,2]
        assert len(set(item[1] for item in seen))>1, "Camera did not update after motor action"
        directory=Path(result["episode_dir"])
        rows=[json.loads(x) for x in (directory/"trajectory.jsonl").read_text().splitlines()]
        assert len(rows)==3 and all(r["execution_valid"] for r in rows)
        assert all(rows[i]["next_images"]==rows[i+1]["images"] for i in range(2))
        assert all(rows[i]["next_robot_state"]==rows[i+1]["robot_state"] for i in range(2))
        reader=imageio.get_reader(str(directory/"simulation.mp4"))
        try:
            count=reader.count_frames()
            assert count==4
            assert reader.get_data(0).shape[:2]==(256,512)
        finally: reader.close()
        print(json.dumps({"validated":True,"real_environment_steps":3,"video_frames":count,
            "external_astra_calls":0,"task_success_claim":False,"directory":str(directory)},indent=2))
    finally:
        agent.close()
        env.close()

if __name__=="__main__": main()
