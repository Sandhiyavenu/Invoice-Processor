from flask import Flask, request, jsonify, send_file, render_template, session, redirect, url_for
from flask_cors import CORS
import pdfplumber
from PIL import Image
import pytesseract
import os
import pandas as pd
import re
import json
from langchain_groq import ChatGroq
import time
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import zipfile
from datetime import datetime
import tempfile
import shutil
from functools import wraps
import easyocr

reader = easyocr.Reader(['en'], gpu=False)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')
CORS(app)

# Initialize ChatGroq model
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY", "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")
)

# Tesseract path (adjust if needed)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Folders - All uploaded files will be stored here
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
OUTPUT_FOLDER = os.path.join(os.getcwd(), "agent", "output")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Simple user database (replace with real database in production)
USERS = {
    'admin': 'admin123',
    'user': 'user123'
}

def extract_text_from_image(image_path):
    # Use EasyOCR to extract text from a single image
    results = reader.readtext(image_path, detail=0)  # detail=0 returns text only
    text = "\n".join(results)
    return text


def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

@app.route('/login')
def login():
    if 'logged_in' in session:
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/api/login', methods=['POST'])
def api_login():
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')

        if username in USERS and USERS[username] == password:
            session['logged_in'] = True
            session['username'] = username
            return jsonify({
                'success': True,
                'message': 'Login successful',
                'token': 'dummy-token-' + username  # Replace with real JWT in production
            })
        else:
            return jsonify({'success': False, 'error': 'Invalid username or password'}), 401
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/api/verify', methods=['POST'])
def verify_token():
    if 'logged_in' in session:
        return jsonify({'success': True})
    return jsonify({'success': False}), 401

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/invoices')
@login_required
def invoice_list():
    return render_template('invoice_list.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        unique_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(file_path)

        try:
            # Extract text
            text = ""
            if file_path.lower().endswith((".jpg", ".jpeg", ".png")):
                print("Extracting text from image using OCR...")
                text = extract_text_from_image(file_path)
            elif file_path.lower().endswith(".pdf"):
                print("Extracting text from PDF...")
                text = extract_text_from_pdf(file_path)

                # OCR fallback for scanned PDFs
                if not text.strip():
                    print("No text found in PDF. Using OCR on pages...")
                    with pdfplumber.open(file_path) as pdf:
                        for i, page in enumerate(pdf.pages):
                            image = page.to_image(resolution=300)
                            ocr_text = pytesseract.image_to_string(image.original)
                            text += f"\n--- Page {i+1} OCR ---\n{ocr_text}"

            if not text.strip():
                # Don't delete the file even if OCR fails
                return jsonify({'error': 'No text found in file even after OCR'}), 400

            print("Text extracted. Sending to ChatGroq for structured Excel...")

            # Prompt LLM for structured JSON
            prompt = f"""
            Extract structured invoice data from the text below and return ONLY JSON.
            Each item must be a separate element in "Items" list.
            Include these fields:

            InvoiceNo, InvoiceDate, BuyerName, BuyerAddress, BuyerGST, SellerName, SellerAddress, SellerGST, Items[], Subtotal, SGST, CGST, IGST, Tax, GrandTotal

            For Items[], include:
            Description, Quantity, UnitPrice, Total

            For GST fields:
            - SGST: State GST @ 9%
            - CGST: Central GST @ 9%
            - IGST: Integrated GST @ 18%
            - Tax: Total tax amount (sum of applicable GST)

            Text:
            {text}
            """

            resp = llm.invoke(prompt)
            llm_text = resp.content.strip()

            # Extract JSON safely
            json_match = re.search(r'\{.*\}', llm_text, re.DOTALL)
            if not json_match:
                return jsonify({'error': 'No JSON found in LLM response'}), 500

            json_str = json_match.group()
            try:
                invoice_data = json.loads(json_str)
            except Exception as e:
                return jsonify({'error': f'Failed to parse JSON: {str(e)}'}), 500

            # Flatten items for Excel
            rows = []
            for item in invoice_data.get("Items", []):
                row = {
                    "InvoiceNo": invoice_data.get("InvoiceNo", ""),
                    "InvoiceDate": invoice_data.get("InvoiceDate", ""),
                    "BuyerName": invoice_data.get("BuyerName", ""),
                    "BuyerAddress": invoice_data.get("BuyerAddress", ""),
                    "BuyerGST": invoice_data.get("BuyerGST", ""),
                    "SellerName": invoice_data.get("SellerName", ""),
                    "SellerAddress": invoice_data.get("SellerAddress", ""),
                    "SellerGST": invoice_data.get("SellerGST", ""),
                    "Description": item.get("Description", ""),
                    "Quantity": item.get("Quantity", ""),
                    "UnitPrice": item.get("UnitPrice", ""),
                    "Total": item.get("Total", ""),
                    "Subtotal": invoice_data.get("Subtotal", ""),
                    "SGST": invoice_data.get("SGST", ""),
                    "CGST": invoice_data.get("CGST", ""),
                    "IGST": invoice_data.get("IGST", ""),
                    "Tax": invoice_data.get("Tax", ""),
                    "GrandTotal": invoice_data.get("GrandTotal", "")
                }
                rows.append(row)

            if not rows:
                return jsonify({'error': 'No items found in invoice JSON'}), 400

            df = pd.DataFrame(rows)

            # Create a better filename
            invoice_no = invoice_data.get("InvoiceNo", "").replace("/", "-").replace("\\", "-")
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")

            if invoice_no:
                output_filename = f"INV_{invoice_no}_{date_str}.xlsx"
            else:
                output_filename = f"Invoice_{date_str}.xlsx"

            output_excel = os.path.join(OUTPUT_FOLDER, output_filename)
            df.to_excel(output_excel, index=False)

            # Store invoice data in session with reference to uploaded file
            if 'processed_invoices' not in session:
                session['processed_invoices'] = []

            session['processed_invoices'].append({
                'invoiceNo': invoice_data.get('InvoiceNo', 'N/A'),
                'invoiceDate': invoice_data.get('InvoiceDate', 'N/A'),
                'buyerName': invoice_data.get('BuyerName', 'N/A'),
                'buyerAddress': invoice_data.get('BuyerAddress', 'N/A'),
                'buyerGST': invoice_data.get('BuyerGST', 'N/A'),
                'sellerName': invoice_data.get('SellerName', 'N/A'),
                'sellerAddress': invoice_data.get('SellerAddress', 'N/A'),
                'sellerGST': invoice_data.get('SellerGST', 'N/A'),
                'subtotal': invoice_data.get('Subtotal', '0.00'),
                'sgst': invoice_data.get('SGST', '0.00'),
                'cgst': invoice_data.get('CGST', '0.00'),
                'igst': invoice_data.get('IGST', '0.00'),
                'tax': invoice_data.get('Tax', '0.00'),
                'grandTotal': invoice_data.get('GrandTotal', '0.00'),
                'items': invoice_data.get('Items', []),
                'excelFile': output_filename,
                'uploadedFile': unique_filename,  # Reference to uploaded file
                'fileType': 'image' if file_path.lower().endswith(('.jpg', '.jpeg', '.png')) else 'pdf'
            })
            session.modified = True

            return jsonify({
                'success': True,
                'message': 'Invoice processed successfully',
                'data': invoice_data,
                'excel_file': output_filename,
                'uploaded_file': unique_filename
            })

        except Exception as e:
            # Don't delete file on error - keep for debugging
            print(f"Error processing file: {str(e)}")
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type. Only PDF, PNG, JPG, and JPEG are allowed'}), 400

@app.route('/api/download/<filename>')
def download_file(filename):
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/files')
def list_files():
    try:
        files = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith('.xlsx')]
        files_with_time = []
        for f in files:
            file_path = os.path.join(OUTPUT_FOLDER, f)
            mod_time = os.path.getmtime(file_path)
            files_with_time.append({
                'name': f,
                'modified': mod_time
            })
        # Sort by modification time, newest first
        files_with_time.sort(key=lambda x: x['modified'], reverse=True)
        return jsonify({'files': files_with_time})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/uploaded-files')
@login_required
def list_uploaded_files():
    """List all uploaded source files"""
    try:
        files = [f for f in os.listdir(UPLOAD_FOLDER) if allowed_file(f)]
        files_with_info = []
        for f in files:
            file_path = os.path.join(UPLOAD_FOLDER, f)
            mod_time = os.path.getmtime(file_path)
            file_size = os.path.getsize(file_path)
            files_with_info.append({
                'name': f,
                'modified': mod_time,
                'size': file_size,
                'type': f.rsplit('.', 1)[1].lower()
            })
        # Sort by modification time, newest first
        files_with_info.sort(key=lambda x: x['modified'], reverse=True)
        return jsonify({'files': files_with_info})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/view-uploaded/<filename>')
@login_required
def view_uploaded_file(filename):
    """View uploaded source files"""
    try:
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            mimetype = 'image/jpeg' if filename.lower().endswith(('.jpg', '.jpeg')) else 'image/png'
        else:
            mimetype = 'application/pdf'

        from flask import Response
        with open(file_path, 'rb') as f:
            file_data = f.read()

        response = Response(file_data, mimetype=mimetype)
        response.headers['Content-Disposition'] = 'inline'
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/update', methods=['POST'])
def update_invoice():
    try:
        data = request.json
        invoice_data = data.get('data')
        original_file = data.get('original_file')

        if not invoice_data:
            return jsonify({'error': 'No data provided'}), 400

        # Flatten items for Excel
        rows = []
        for item in invoice_data.get("Items", []):
            row = {
                "InvoiceNo": invoice_data.get("InvoiceNo", ""),
                "InvoiceDate": invoice_data.get("InvoiceDate", ""),
                "BuyerName": invoice_data.get("BuyerName", ""),
                "BuyerAddress": invoice_data.get("BuyerAddress", ""),
                "BuyerGST": invoice_data.get("BuyerGST", ""),
                "SellerName": invoice_data.get("SellerName", ""),
                "SellerAddress": invoice_data.get("SellerAddress", ""),
                "SellerGST": invoice_data.get("SellerGST", ""),
                "Description": item.get("Description", ""),
                "Quantity": item.get("Quantity", ""),
                "UnitPrice": item.get("UnitPrice", ""),
                "Total": item.get("Total", ""),
                "Subtotal": invoice_data.get("Subtotal", ""),
                "SGST": invoice_data.get("SGST", ""),
                "CGST": invoice_data.get("CGST", ""),
                "IGST": invoice_data.get("IGST", ""),
                "Tax": invoice_data.get("Tax", ""),
                "GrandTotal": invoice_data.get("GrandTotal", "")
            }
            rows.append(row)

        if not rows:
            return jsonify({'error': 'No items provided'}), 400

        # Generate new Excel file
        df = pd.DataFrame(rows)

        # Create a better filename
        invoice_no = invoice_data.get("InvoiceNo", "").replace("/", "-").replace("\\", "-")
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        if invoice_no:
            output_filename = f"INV_{invoice_no}_{date_str}_updated.xlsx"
        else:
            output_filename = f"Invoice_{date_str}_updated.xlsx"

        output_excel = os.path.join(OUTPUT_FOLDER, output_filename)
        df.to_excel(output_excel, index=False)

        return jsonify({
            'success': True,
            'message': 'Invoice updated successfully',
            'excel_file': output_filename
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/session-invoices')
@login_required
def get_session_invoices():
    invoices = session.get('processed_invoices', [])
    return jsonify({'invoices': invoices})

@app.route('/api/view-invoice/<filename>')
@login_required
def view_invoice_file(filename):
    """View files from OUTPUT folder (for backward compatibility)"""
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            mimetype = 'image/jpeg' if filename.lower().endswith(('.jpg', '.jpeg')) else 'image/png'
        else:
            mimetype = 'application/pdf'

        from flask import Response
        with open(file_path, 'rb') as f:
            file_data = f.read()

        response = Response(file_data, mimetype=mimetype)
        response.headers['Content-Disposition'] = 'inline'
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/download_multiple', methods=['POST'])
def download_multiple():
    try:
        data = request.json
        files_to_download = data.get('files', [])

        if not files_to_download:
            return jsonify({'error': 'No files selected'}), 400

        # Create unique folder name with date and timestamp
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        zip_filename = f"Invoices_{date_str}.zip"
        zip_path = os.path.join(OUTPUT_FOLDER, zip_filename)

        # Create ZIP file
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for filename in files_to_download:
                file_path = os.path.join(OUTPUT_FOLDER, filename)
                if os.path.exists(file_path):
                    # Add file to ZIP with just the filename (no path)
                    zipf.write(file_path, filename)
                else:
                    return jsonify({'error': f'File not found: {filename}'}), 404

        # Send the ZIP file
        return send_file(
            zip_path,
            as_attachment=True,
            download_name=zip_filename,
            mimetype='application/zip'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.json
    # Validate and create user
    # Hash password
    # Store in database
    # Return success with token
    return jsonify({
        'success': True,
        'token': 'your_jwt_token',
        'redirect': '/signup'  # or '/' for dashboard
    })

@app.route('/api/download_multiple_invoices', methods=['POST'])
def download_multiple_invoices():
    try:
        data = request.json
        invoices = data.get('invoices', [])

        if not invoices:
            return jsonify({'error': 'No invoices selected'}), 400

        # Collect all invoice data
        all_rows = []

        for invoice in invoices:
            # Process each invoice's items
            items = invoice.get('items', [])

            if items:
                for item in items:
                    row = {
                        "InvoiceNo": invoice.get("invoiceNo", ""),
                        "InvoiceDate": invoice.get("invoiceDate", ""),
                        "BuyerName": invoice.get("buyerName", ""),
                        "BuyerAddress": invoice.get("buyerAddress", ""),
                        "BuyerGST": invoice.get("buyerGST", ""),
                        "SellerName": invoice.get("sellerName", ""),
                        "SellerAddress": invoice.get("sellerAddress", ""),
                        "SellerGST": invoice.get("sellerGST", ""),
                        "Description": item.get("Description", ""),
                        "Quantity": item.get("Quantity", ""),
                        "UnitPrice": item.get("UnitPrice", ""),
                        "Total": item.get("Total", ""),
                        "Subtotal": invoice.get("subtotal", ""),
                        "SGST": invoice.get("sgst", ""),
                        "CGST": invoice.get("cgst", ""),
                        "IGST": invoice.get("igst", ""),
                        "Tax": invoice.get("tax", ""),
                        "GrandTotal": invoice.get("grandTotal", "")
                    }
                    all_rows.append(row)
            else:
                # If no items, add invoice info without item details
                row = {
                    "InvoiceNo": invoice.get("invoiceNo", ""),
                    "InvoiceDate": invoice.get("invoiceDate", ""),
                    "BuyerName": invoice.get("buyerName", ""),
                    "BuyerAddress": invoice.get("buyerAddress", ""),
                    "BuyerGST": invoice.get("buyerGST", ""),
                    "SellerName": invoice.get("sellerName", ""),
                    "SellerAddress": invoice.get("sellerAddress", ""),
                    "SellerGST": invoice.get("sellerGST", ""),
                    "Description": "",
                    "Quantity": "",
                    "UnitPrice": "",
                    "Total": "",
                    "Subtotal": invoice.get("subtotal", ""),
                    "SGST": invoice.get("sgst", ""),
                    "CGST": invoice.get("cgst", ""),
                    "IGST": invoice.get("igst", ""),
                    "Tax": invoice.get("tax", ""),
                    "GrandTotal": invoice.get("grandTotal", "")
                }
                all_rows.append(row)

        if not all_rows:
            return jsonify({'error': 'No data to export'}), 400

        # Create DataFrame
        df = pd.DataFrame(all_rows)

        # Generate filename
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"Combined_Invoices_{date_str}.xlsx"
        output_excel = os.path.join(OUTPUT_FOLDER, output_filename)

        # Write to Excel
        df.to_excel(output_excel, index=False)

        # Send the file
        return send_file(
            output_excel,
            as_attachment=True,
            download_name=output_filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)