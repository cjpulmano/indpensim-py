"""Streamlit UI for the indpensim Recipe layer.

Gated behind the ``ui`` optional extra. Importing any module in this
package without ``streamlit`` / ``plotly`` installed will raise the
usual ``ImportError`` from the missing dependency.

Entry point:
    streamlit run indpensim/ui/streamlit_app.py

Pages:
    01_authoring.py — form-based recipe authoring (CRUD + JSON I/O).
    02_visualize.py — load a Recipe JSON, plot the timeline.
"""
