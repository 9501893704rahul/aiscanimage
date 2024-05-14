import os
from flask import Flask, render_template, request, redirect, url_for
import json
import time
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Read from environment variables
subscription_key = os.getenv('AZURE_COMPUTER_VISION_SUBSCRIPTION_KEY')
endpoint = os.getenv('AZURE_COMPUTER_VISION_ENDPOINT')
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

text_analytics_subscription_key = os.getenv('AZURE_TEXT_ANALYTICS_SUBSCRIPTION_KEY')
text_analytics_endpoint = os.getenv('AZURE_TEXT_ANALYTICS_ENDPOINT')
text_analytics_client = TextAnalyticsClient(endpoint=text_analytics_endpoint, credential=AzureKeyCredential(text_analytics_subscription_key))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_image(url):
    read_response = computervision_client.read(url, raw=True)
    operation_location = read_response.headers["Operation-Location"]
    operation_id = operation_location.split("/")[-1]

    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status.lower() not in ['notstarted', 'running']:
            break
        time.sleep(1)

    if read_result.status == 'succeeded':
        return [{'text': line.text} for text_result in read_result.analyze_result.read_results for line in text_result.lines]
    else:
        return []

def recognize_entities(texts):
    max_batch_size = 5
    results = []

    for i in range(0, len(texts), max_batch_size):
        batch = [{"id": str(i + j), "text": text} for j, text in enumerate(texts[i:i+max_batch_size]) if text.strip()]
        if not batch:
            continue

        try:
            response = text_analytics_client.recognize_entities(documents=batch)
            for doc in response:
                if doc.is_error:
                    print(f"Document error: {doc.error}")
                    continue

                doc_id = int(doc.id)
                if doc_id >= i and doc_id < i + len(batch):
                    actual_index = doc_id - i
                    results.append({
                        'text': batch[actual_index]['text'],
                        'entities': [{'text': entity.text, 'category': entity.category, 'confidence_score': entity.confidence_score} for entity in doc.entities]
                    })
                else:
                    print(f"Error: Document ID {doc_id} is out of range for the batch")
        except Exception as e:
            print(f"Error in entity recognition for batch starting at index {i}: {e}")

    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract', methods=['POST'])
def extract():
    if 'image_file' not in request.files and 'image_url' not in request.form:
        return redirect(request.url)

    if 'image_file' in request.files:
        image_file = request.files['image_file']
        if image_file and allowed_file(image_file.filename):
            filename = secure_filename(image_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(filepath)

            with open(filepath, 'rb') as img:
                read_response = computervision_client.read_in_stream(img, raw=True)
            operation_location = read_response.headers["Operation-Location"]
            operation_id = operation_location.split("/")[-1]

            while True:
                read_result = computervision_client.get_read_result(operation_id)
                if read_result.status.lower() not in ['notstarted', 'running']:
                    break
                time.sleep(1)

            if read_result.status == 'succeeded':
                text_results = [{'text': line.text} for text_result in read_result.analyze_result.read_results for line in text_result.lines]
                extracted_texts = [result['text'] for result in text_results if result['text'].strip()]
                entity_results = recognize_entities(extracted_texts)
                os.remove(filepath)
                return render_template('results.html', entity_results=entity_results)
    
    elif 'image_url' in request.form:
        image_url = request.form['image_url']
        text_results = extract_text_from_image(image_url)
        extracted_texts = [result['text'] for result in text_results if result['text'].strip()]
        entity_results = recognize_entities(extracted_texts)
        return render_template('results.html', entity_results=entity_results)

    return redirect(request.url)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run()
