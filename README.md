# README for FAQ Chatbot Project

## Project Overview
This FAQ Chatbot is designed to provide automated responses to common questions in several categories, including WiFi issues, email issues, laptop issues, and SAP issues. The chatbot utilizes a pre-trained DistilBERT model to process user queries and match them with the most relevant answers from a set of predefined FAQs.

## Features
- **Multiple Categories:** Handles queries related to WiFi, Email, Laptop, and SAP issues.
- **Intelligent Matching:** Uses SentenceTransformer's DistilBERT model for semantic similarity.
- **Threshold-based Response:** Provides a threshold score for each category to ensure relevant responses.
- **Fallback Option:** Offers contact information for unresolved queries.

## Installation

### Prerequisites
- Python 3.6+
- Flask
- Pandas
- Sentence Transformers

### Setup
1. Clone the repository to your local machine.
2. Install the required dependencies:
   ```bash
   pip install flask pandas sentence-transformers flask-cors
   ```
3. Place the CSV files (`wifi_issues.csv`, `Email_Issues.csv`, `Laptop_Issues.csv`, `Sap_Issues.csv`) in the project directory.

## Running the Application
1. Navigate to the project directory.
2. Run the Flask app:
   ```bash
   python app.py
   ```
3. The service will start on `localhost` at port `8080`.

## Usage
Send a POST request to the appropriate endpoint with a JSON payload containing the user query. Example endpoints:
- `/answer/wifi`
- `/answer/email`
- `/answer/laptop`
- `/answer/sap`

## Response Format
The API returns a JSON object with the key `answer` containing the response to the user query.


## Contributing
Contributions to improve the chatbot are welcome. Please follow the standard procedure for contributing to a GitHub project.

