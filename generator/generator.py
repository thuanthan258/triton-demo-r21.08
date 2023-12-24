from pydantic import BaseModel
from typing import List, Dict, Any
import json
from pathlib import Path
import os

from loguru import logger


class InputBlockConfig(BaseModel):
    """Input block configuration"""

    input_cols: List[str]
    name_mapping: Dict[str, str]


class BlockConfig(BaseModel):
    """Full block configuration"""

    inputs: Dict[str, InputBlockConfig]
    output_cols: List[str]
    max_historical_days: int
    env_var: Dict[str, Any]


class TemplateHandler:
    def __init__(self, output_dir: str, execution_id: str) -> None:
        self.abs_path = Path(__file__).parent.absolute()

        self.config_template_path = os.path.join(
            self.abs_path, "template/config_template.pbtxt.txt"
        )
        self.config_template = self._read_template(self.config_template_path)

        self.model_template_path = os.path.join(
            self.abs_path, "template/model_template.py.txt"
        )
        self.model_template = self._read_template(self.model_template_path)

        self.execution_path = os.path.join(output_dir, f"{execution_id}")
        if not os.path.exists(self.execution_path):
            Path(self.execution_path).mkdir(parents=True, exist_ok=True)

        self.version_path = os.path.join(self.execution_path, "1")
        if not os.path.exists(self.version_path):
            Path(self.version_path).mkdir(parents=True, exist_ok=True)

        self.config_path = os.path.join(self.version_path, "config")
        if not os.path.exists(self.config_path):
            Path(self.config_path).mkdir(parents=True, exist_ok=True)

        self.execution_plan_path = os.path.join(self.config_path, "execution_plan.json")
        self.execution_plan = {}
        if not os.path.isfile(self.execution_plan_path):
            Path(self.execution_plan_path).touch()
        else:
            with open(self.execution_plan_path, "r") as f:
                self.execution_plan = json.load(f)

    def _read_template(self, template_path: str) -> str:
        with open(template_path, "r") as f:
            template = f.read()
        return template

    def update_config_template(self, field: str, value: str) -> None:
        self.config_template = self.config_template.replace(field, value)

    def write_config_template(
        self,
    ) -> None:
        with open(os.path.join(self.execution_path, "config.pbtxt"), "w") as f:
            f.write(self.config_template)

    def copy_file_to_execution_path(
        self, file_path: str, new_name: str, block_id: str
    ) -> None:
        os.system(
            f'cp {file_path} {os.path.join(self.config_path, f"{block_id}/{new_name}")}'
        )

    def update_execution_plan(self, block_id: str, block_config: BlockConfig) -> None:
        self.execution_plan[block_id] = block_config.dict()

    def write_execution_plan(self) -> None:
        with open(self.execution_plan_path, "w") as f:
            json.dump(self.execution_plan, f)

    def write_model_template(self):
        with open(os.path.join(self.version_path), "model.py") as f:
            f.write(self.model_template)


if __name__ == "__main__":
    result = {"topo_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    block_1_input_1 = InputBlockConfig(
        input_cols=["a", "b"],
        name_mapping={"a": "a", "b": "b"},
    )
    block_1_input_2 = InputBlockConfig(
        input_cols=["c", "d"],
        name_mapping={"c": "c", "d": "d"},
    )
    block_1 = BlockConfig(
        inputs={"input_1": block_1_input_1, "input_2": block_1_input_2},
        output_cols=["a", "b", "c", "d"],
        max_historical_days=10,
        env_var={"topo_ids": "topo_ids"},
    )
    result["block_1"] = block_1.model_dump()
    result = json.dumps(result)
    logger.info(result)
