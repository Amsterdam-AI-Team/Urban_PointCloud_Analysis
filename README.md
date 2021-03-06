# Extracting urban street furniture from labeled point clouds

This repository contains methods for the **automatic extraction of urban street furniture** from labeled PointClouds. It serves as a post-processing toolbox for our [data fusion labeling approach](https://github.com/Amsterdam-AI-Team/Urban_PointCloud_Processing), but it can also be used for point clouds that were labeled manually or by semantic segmentation algorithms, with minor modifications.

This is very much work in progress: for now, we only provide code to extract the location of pole-like objects. Further methods to extract other types of street furniture will follow. For more background on this project, please have a look at our accompanying [blog post](https://amsterdamintelligence.com/posts/locating-street-lights-in-point-clouds-using-deep-learning).

<figure align="center">
  <img
  src="media/examples/extracted_pole.png"
  alt="Example: visualization of one extracted pole.">
  <figcaption><b>Example:</b> visualization of one extracted pole.</figcaption>
</figure>

---

## Folder Structure

 * [`datasets`](./datasets) _Demo dataset to get started_
   * [`ahn`](./datasets/ahn) _AHN elevation data_
   * [`pointcloud`](./datasets/pointcloud) _Example labeled urban point clouds_
 * [`media/examples`](./media/examples) _Visuals_
 * [`notebooks`](./notebooks) _Jupyter notebook tutorials_
 * [`src/upc_analysis`](./src/upc_analysis) _Python source code_

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
    pip install -r requirements-notebooks.txt
    ```

3. Additionally, install our [Urban PointCloud Processing](https://github.com/Amsterdam-AI-Team/Urban_PointCloud_Processing) package:
    ```bash
    pip install https://github.com/Amsterdam-AI-Team/Urban_PointCloud_Processing/releases/download/v0.1/upcp-0.1-py3-none-any.whl
    ```

4. Finally, install `cccorelib` and `pycc` by following the [instructions on their GitHub page](https://github.com/tmontaigu/CloudCompare-PythonPlugin/blob/master/docs/building.rst#building-as-independent-wheels). Please note, these two packages are not available on the Python Package Index (PyPi).

5. Check out the [notebooks](notebooks) for a demonstration.

---

## Usage

We provide tutorial [notebooks](notebooks) that demonstrate how the tools can be used. Labeled example point clouds are provided to get started.

---

This repository was created by [Amsterdam Intelligence](https://amsterdamintelligence.com/) for the City of Amsterdam.
