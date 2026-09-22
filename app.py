"""
Application Flask pour le comparateur YOLO.
"""

import os
import uuid
import json
from pathlib import Path

from flask import (Flask, render_template, request, jsonify,
                   send_from_directory, url_for)
from werkzeug.utils import secure_filename

from yolo_benchmark import YOLOBenchmark

# ─── Configuration ───────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB max
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp', 'tiff'}

# Créer les dossiers
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Instance du benchmark
benchmark = YOLOBenchmark(
    upload_folder=app.config['UPLOAD_FOLDER'],
    results_folder=app.config['RESULTS_FOLDER']
)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ─── Routes ──────────────────────────────────────────────────
@app.route('/')
def index():
    """Page principale."""
    models = benchmark.get_available_models()
    return render_template('index.html', models=models)


@app.route('/api/models', methods=['GET'])
def get_models():
    """API: Liste des modèles disponibles."""
    return jsonify(benchmark.get_available_models())


@app.route('/api/benchmark', methods=['POST'])
def run_benchmark():
    """API: Lancer un benchmark."""
    # Vérifier l'image
    if 'image' not in request.files:
        return jsonify({"error": "Aucune image fournie"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Aucun fichier sélectionné"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Format non supporté. Formats acceptés: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    # Récupérer les modèles sélectionnés
    selected_models = request.form.getlist('models')
    if not selected_models:
        # Fallback: essayer de parser depuis un champ JSON
        models_json = request.form.get('models_json', '[]')
        try:
            selected_models = json.loads(models_json)
        except json.JSONDecodeError:
            pass

    if not selected_models:
        return jsonify({"error": "Aucun modèle sélectionné"}), 400

    # Seuil de confiance
    conf_threshold = float(request.form.get('confidence', 0.25))

    # Sauvegarder l'image
    ext = file.filename.rsplit('.', 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(image_path)

    # Lancer le benchmark
    try:
        results = benchmark.benchmark_multiple(
            model_names=selected_models,
            image_path=image_path,
            conf_threshold=conf_threshold
        )

        # Générer les images annotées
        for res in results:
            if res['success'] and res['detections']:
                annotated_name = benchmark.generate_annotated_image(
                    image_path, res['detections'], res['model_name']
                )
                res['annotated_image'] = annotated_name

        # Sauvegarder les résultats
        results_file = benchmark.save_results(results, f"bench_{unique_name}.json")

        response = {
            "success": True,
            "image": unique_name,
            "results": results,
            "results_file": results_file
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Servir les images uploadées."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/results/<filename>')
def serve_result(filename):
    """Servir les fichiers de résultats."""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)


# ─── Main ────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print(" YOLO Benchmark Comparator")
    print("=" * 60)
    print(f" Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f" Results folder: {app.config['RESULTS_FOLDER']}")
    print(f" URL: http://localhost:5000")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)