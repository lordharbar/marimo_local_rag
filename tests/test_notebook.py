import marimo

__generated_with = "0.10.0"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __(mo):
    mo.md("# Test Notebook")
    return


@app.cell
def __(mo):
    button = mo.ui.button(label="Click me")
    return button,


@app.cell
def __(button):
    button
    return


@app.cell
def __(button, mo):
    if button.value > 0:
        result = mo.md(f"Button clicked {button.value} times")
    else:
        result = mo.md("Button not clicked yet")
    return result,


@app.cell
def __(result):
    result
    return


if __name__ == "__main__":
    app.run()
