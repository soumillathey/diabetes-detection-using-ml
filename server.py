import os
from api.index import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"Starting server on http://localhost:{port}")
    app.run(debug=False, host='0.0.0.0', port=port)
