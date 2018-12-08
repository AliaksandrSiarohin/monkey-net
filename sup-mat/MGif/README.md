## Evaluation

There are 2 versions of MGif dataset: **raw** and **processed**.

#### Raw

The raw dataset is composed of the *.gif* files. Files is manually downloaded using google search, we use the following search phrase "cat cycle gif animation". And replace the cat with different animals such as dog, elephant, camel and so on. The original names is preserved. The dataset can be [downloaded](https://yadi.sk/d/VBFiPs95yO7sMA).

#### Processed

In processed dataset bg is manually removed using the tool ```bg_removal_tool.py```.
Usage:
```
python bg_removal_tool.py /folder/with/raw/gifs /folder/for/processed/gifs /folder/for/bad/gifs/which/you/want/to/exclude
```
All the gifs is resized to 256 x 256. The files is renamed, correspondence between **raw** and **processed** could be established via *mapping.txt* file. The dataset can be [downloaded](https://yadi.sk/d/VBFiPs95yO7sMA).

