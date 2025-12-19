import gradio as gr
from transformers import T5Tokenizer, T5ForConditionalGeneration
import zipfile

# Hugging Faceの事前学習済みモデルを使う
MODEL_NAME = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

def build_sb3(json_code, output_path="output.sb3"):
    with open("project.json", "w") as f:
        f.write(json_code)
    with zipfile.ZipFile(output_path, "w") as z:
        z.write("project.json")
    return output_path

def generate_sb3(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_length=512)
    json_code = tokenizer.decode(output[0], skip_special_tokens=True)
    sb3_path = build_sb3(json_code)
    return json_code, sb3_path

gr.Interface(
    fn=generate_sb3,
    inputs=gr.Textbox(label="プロンプトを入力（例：猫がジャンプするゲームを作って）"),
    outputs=[
        gr.Textbox(label="生成されたproject.json"),
        gr.File(label="ダウンロード用sb3ファイル")
    ],
    title="プロンプトからScratchプロジェクトを生成",
    description="自然言語で指示を出すと、Scratchのsb3ファイルを生成します。"
).launch()
