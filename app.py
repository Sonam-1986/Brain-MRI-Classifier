"""
MRI Brain Tumor Analyzer - Flask Application
"""
import os
import json
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from model.tumor_classifier import MRITumorClassifier

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024   # 10 MB per file
app.config['UPLOAD_FOLDER'] = 'uploads'

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
classifier = MRITumorClassifier()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'files' not in request.files:
        return jsonify({'success': False, 'error': 'No files provided.'}), 400

    files = request.files.getlist('files')

    if len(files) == 0:
        return jsonify({'success': False, 'error': 'No files selected.'}), 400

    if len(files) > 4:
        return jsonify({'success': False, 'error': 'Maximum 4 files allowed.'}), 400

    results = []
    for f in files:
        if f.filename == '':
            continue

        if not allowed_file(f.filename):
            results.append({
                'filename': f.filename,
                'success':  False,
                'error':    f'File type not supported. Use JPG or PNG.',
            })
            continue

        image_bytes = f.read()
        result = classifier.analyze(image_bytes)
        result['filename'] = secure_filename(f.filename)
        results.append(result)

    return jsonify({'results': results})


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    host = '0.0.0.0'
    print("=" * 55)
    print("  MRI Brain Tumor Analyzer")
    print(f"  Open: http://127.0.0.1:{port}")
    print("=" * 55)
    app.run(debug=False, host=host, port=port)
