from flask import Flask
from flask import request, stream_with_context
import uuid
from dotenv import load_dotenv
from qe_controller import QueryEngineController

from common_lib.utils.load_config import load_config
from utils.stream_utils import stream_generator

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

global qe_ctrl
qe = None
@app.route("/query", methods=["POST"])
def query():
    content = request.json
    id = uuid.uuid4().hex[:10]
    # query = content.get('query')
    query = request.json['messages'][-1]['content']
    if query is None:
        query = " "
    response = qe.query(query)
    return app.response_class(stream_with_context(stream_generator(id, response)))

if __name__ == "__main__":
    load_dotenv()

    config = load_config()
    qe_ctrl = QueryEngineController(config)
    qe = qe_ctrl.get_query_engine()

    app.run(host="0.0.0.0", port=5001)

