from typing import List, Any
Dialog = List[List[dict]]
class Llama:
    @classmethod
    def build(cls, *args, **kwargs):
        return cls()
    def chat_completion(self, dialogs: Dialog, max_gen_len=512, temperature=0.6, top_p=0.9):
        # Step1 only needs a list with a dict containing 'generation'->'content'
        return [{"generation": {"content": ""}}]
