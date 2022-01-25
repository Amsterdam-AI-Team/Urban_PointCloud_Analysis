# Urban PointCloud Analysis

This repository contains methods for the **automatic extraction of urban street furniture** from labeled PointClouds.

---

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/Amsterdam-AI-Team/Urban_PointCloud_Analysis.git
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    # optional requirements for Jupyter notebooks:
    pip install -r requirement-notebooks.txt
    ```

3. Additionally, install our [Urban PointCloud Processing](https://github.com/Amsterdam-AI-Team/Urban_PointCloud_Processing) package using the wheel provided in this repository:
    ```bash
    pip install upcp-0.1a0-py3-none-any.whl
    ```

4. Finally, install `cccorelib` and `pycc` by following the [instructions on their GitHub page](https://github.com/tmontaigu/CloudCompare-PythonPlugin/blob/master/docs/building.rst#building-as-independent-wheels).

5. Check out the [notebooks](notebooks) for a demonstration.
