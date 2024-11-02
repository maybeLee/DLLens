## Structure of `./data` directory
```angular2html
./
├── README.md
├── tf_api_list.txt
├── tf_filtered_apis.txt
├── tf_skip_pref.json
├── torch_api_list.txt
├── torch_filtered_apis.txt
├── torch_skip_pref.json
└── working_dir/rq1/counterpart
```

- `./api.key` contains the key used for accessing the OpenAI's gpt-turbo-3.5 service.
- `./api_implementations` contains the implementations of the API in PyTorch and TensorFlow.
- `./tf_api_100samples.txt` and `./torch_api_100samples.txt` 
  - contain sampled 100 TensorFlow and PyTorch, respectively.
  - These sampled APIs are used in RQ4.
- `./tf_api_list.txt` and `./torch_api_list.txt`
  - contain the total list of TensorFlow APIs (3249) and PyTorch APIs (1574), respectively.
- `./tf_skip_pref.txt` and `./torch_skip_pref.txt`
  - contain the APIs we excluded in our experiment. 
  - These APIs were excluded because they are DL library specific and not related to general DL functionalities.
- `tf_filted_apis.txt` and `torch_filtered_apis.txt`
  - contain the filtered APIs that are used in our experiment.
  - Note that besides excluding APIs based on `*_skip_pref.txt`, 
  - we also exclude APIs that can be called without given input arguments.
  - Finally, 1862 TensorFlow APIs and 1121 PyTorch APIs are included in our experiment.
- `./working_dir/rq1/counterpart` contains the counterparts synthesized by DLLens in RQ1. 
- `./working_dir/rq2/constraints` contains the constraints collected by DLLens in RQ2.
