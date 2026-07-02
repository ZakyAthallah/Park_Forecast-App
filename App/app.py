import argparse
import sys

from flask import Flask, request, render_template
import pandas as pd


def create_web_app():
    app = Flask(__name__)

    def simple_forecast(series, periods=12):
        if len(series) == 0:
            return []
        last = series.iloc[-1]
        return [float(last)] * periods

    @app.route('/', methods=['GET', 'POST'])
    def index():
        forecast = None
        periods = 12
        message = None
        if request.method == 'POST':
            file = request.files.get('file')
            try:
                periods = int(request.form.get('periods', 12))
            except Exception:
                periods = 12

            if file:
                try:
                    df = pd.read_csv(file)
                    num_cols = df.select_dtypes(include=['number']).columns
                    if len(num_cols) == 0:
                        message = 'CSV contains no numeric columns.'
                        forecast = []
                    else:
                        series = df[num_cols[0]]
                        forecast = simple_forecast(series, periods)
                except Exception as e:
                    message = f'Error reading CSV: {e}'
                    forecast = []

        return render_template('index.html', forecast=forecast, periods=periods, message=message)

    return app


def run_gui():
    try:
        from parking_forecast_legacy import App
    except ModuleNotFoundError as e:
        print("Could not import GUI dependencies:", e, file=sys.stderr)
        print("If you want to run the GUI, please install the required optional packages.", file=sys.stderr)
        raise

    gui = App()
    gui.mainloop()


def main():
    parser = argparse.ArgumentParser(
        description="Launch the Forecasting GUI or the Flask web app."
    )
    parser.add_argument(
        '--web', action='store_true', help='Run the Flask web app instead of the GUI.'
    )
    parser.add_argument(
        '--host', default='0.0.0.0', help='Host for web app (when using --web).'
    )
    parser.add_argument(
        '--port', type=int, default=5000, help='Port for web app (when using --web).'
    )
    args = parser.parse_args()

    if args.web:
        print(f"Starting Flask web app at http://{args.host}:{args.port}/")
        app = create_web_app()
        app.run(host=args.host, port=args.port)
    else:
        print("Starting GUI mode. Close the window to exit.")
        try:
            run_gui()
        except ModuleNotFoundError as e:
            missing = getattr(e, 'name', None) or str(e)
            print(f"GUI dependencies missing: {missing}", file=sys.stderr)
            print("Falling back to the Flask web app. Use --web to force web mode.", file=sys.stderr)
            app = create_web_app()
            app.run(host=args.host, port=args.port)


if __name__ == '__main__':
    main()
