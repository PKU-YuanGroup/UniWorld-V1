class Prompter:
    def __init__(self):
        pass

    def get_train_prompt(self, data: list[dict]) -> list[dict]:
        pass

    def __call__(self, data: list[dict], generate_format: bool = True) -> str:
        pass


class Qwen2p5VLPrompter(Prompter):
    def __init__(self):
        self.bos_token = "<|im_start|>"
        self.eos_token = "<|im_end|>"

        self.role_list = ["user", "assistant", "system"]

        self.assistant_role = "assistant"
        self.system_role = "system"
        self.user_role = "user"
        self.default_system_prompt = "You are a helpful assistant."

        self.prompt_template = "{bos_token}{role}\n{prompt}{eos_token}"
        
    def get_train_prompt(self, data: list[dict]) -> list[dict]:
        prompt_list = []
        conversation_length = len(data)
        for idx, item in enumerate(data):
            if item["from"] not in self.role_list:
                raise ValueError(f"Role {item['from']} is not in the role list")

            if item["from"] == self.assistant_role:
                prompt_list.append(
                    {
                        "prompt": f"{self.bos_token}{item['from']}\n",
                        "is_labels": False,
                        "from": item["from"],
                    }
                )
                prompt_list.append(
                    {
                        "prompt": f"{item['value']}{self.eos_token}",
                        "is_labels": True,
                        "from": item["from"],
                    }  # Make it labels
                )
            elif item["from"] == self.system_role or item["from"] == self.user_role:
                prompt_list.append(
                    {
                        "prompt": f"{self.bos_token}{item['from']}\n{item['value']}{self.eos_token}",
                        "is_labels": False,
                        "from": item["from"],
                    }
                )
            else:
                raise ValueError(f"Role {item['from']} is not in the role list")

            if idx != conversation_length - 1:
                prompt_list.append(
                    {"prompt": "\n", "is_labels": False, "from": item["from"]}
                )

        return prompt_list

    def __call__(self, data: list[dict]) -> str:
        prompt_list = []
        for item in data:
            if item["from"] not in self.role_list:
                raise ValueError(f"Role {item['from']} is not in the role list")

            prompt_list.append(
                self.prompt_template.format(
                    bos_token=self.bos_token,
                    role=item["from"],
                    prompt=item["value"],
                    eos_token=self.eos_token,
                )
            )

        prompt_list.append(
            self.prompt_template.format(
                bos_token=self.bos_token,
                role=self.assistant_role,
                prompt="",
                eos_token="",
            )
        )

        return "\n".join(prompt_list)

PROMPT_TYPE = {
    'qwen2p5vl': Qwen2p5VLPrompter, 
    'qwen2p5vl_v1_1': Qwen2p5VLPrompter, 
}