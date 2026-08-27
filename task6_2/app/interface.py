import gradio as gr
import requests


# FastAPI URL


API_URL = "http://127.0.0.1:8000/caption"


# Send image to FastAPI

def generate_caption(image_path):

    if image_path is None:
        return "Please upload an image."

    try:

        # Open the image file
        with open(image_path, "rb") as image_file:

            files = { "image": ("image.jpg", image_file, "image/jpeg")}

            # Send image to FastAPI
            response = requests.post( API_URL, files=files)

        # Check for errors
        response.raise_for_status()

        # Read API response
        result = response.json()

        return result["caption"]

    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to the FastAPI server."

    except requests.exceptions.HTTPError as e:
        return f"API error: {e}"

    except Exception as e:
        return f"Error: {e}"


# UI

with gr.Blocks(title="Image Captioning") as interface:

    gr.Markdown()

    image_input = gr.Image( type="filepath", label="Upload Image")

    generate_button = gr.Button("Generate Caption")

    caption_output = gr.Textbox( label="Generated Caption", lines=3 )

    generate_button.click( fn=generate_caption, inputs=image_input, outputs=caption_output )


# Start Gradio
if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860
    )