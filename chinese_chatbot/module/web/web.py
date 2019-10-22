from flask import Flask
from flask import jsonify
from flask import request, render_template


def create_app(chatbot):
    app = Flask(__name__)

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/chat', methods=['POST'])
    def chat():
        if request.method == 'POST':
            ask = request.form['ask']

            print("get input ask: ", ask)
            answer = chatbot.chat(ask).final_answer

            print()

            return jsonify({"status": "ok", "answer": answer})

    return app


# def main():
#     app = create_app(None)
#     app.run(host=config.host, port=config.port, debug=config.debug)
#
#
# if __name__ == '__main__':
#     main()
