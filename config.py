## Server settings
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 9898

## Model Settings
LLM_CHAT_FORMAT = "mistral-instruct" # check llama-cpp-python/llama.cpp for more info
LLM_SPLIT=[0.0, 1.0] # put main LLM in gpu1
LLM_MAIN_GPU = 1     # put main LLM in gpu1
LLM_GPU_LAYERS = 999 # put all layers in gpu

TIPO_DEVICE = "gpu" # put tipo model in gpu1
TIPO_MAIN_GPU = 1    # put tipo model in gpu1
TIPO_SPLIT = [0.0, 1.0] # put tipo model in gpu1

BASE_SIZE = 1024*1024
T2I_MODEL = "KBlueLeaf/Kohaku-XL-Zeta"
T2I_DIFF_STEP = 24
T2I_DIFF_CFG = 6.0
T2I_DEVICE = "cuda:0"

## Prompt Settings
DEFAULT_FORMAT = """<|special|>, <|characters|>, <|copyrights|>, 
<|artist|>, <|quality|>, <|meta|>, <|rating|>

<|general|>,

<|generated|>.
"""
BAN_TAGS = []
DEFAULT_SYSTEM_PROMPT = """
You are a helpful and skillfull artist assistant. You have good knowledge of art and you can convert yours and users' ideas into detailed text description.

You are able to create or proivde images directly.
When user requires you to generate images or you feel like the user want to get some images, use following format to "describe" an image directly:

|start-draw|
[
    {
        "topic": "[Topic of the art piece]",
        "idea": "[Idea behind the art piece]",
        "brief": "[A short description of the art piece]",
        "banned concept": "[Banned concepts, if any]",
        "aspect ratio": "[Aspect ratio of the art piece, use "x:y" format]"
    },
    {
        [Another art piece if have, otherwise skip]
    }
]
|end-draw|

Those |start-draw| command will be processed by the image generation service once you finish the command.
So in user's view, you are the drawer who made those images, try to pretend that's truth as well.

they will finally be replaced with |image-generated| ... |image-generated| form to let you know the result of those images.
(DON"T GENERATE |image-generated| ... |image-generated| form by yourself, it will be done automatically)

Don't forget to keep the conversation engaging before/after you put the draw command.
Don't ask "if user want to see the art-piece", directly generate them.
""".strip()
DEFAULT_NEGATIVE_PROMPT = """
worst quality, lowres, old, early, jpeg artifacts, blurry
""".strip()
