<h1 align="center"> Attention-Box</h1>
<p align="center">  A tensorflow addon for many types of attention mechanisms</p>


You can use attention toolbox by passing a tensor like :
```python

from attention_box import soft_attention

# query is from embedding layer or lstm or cnn logit

attention_output = soft_attention(query, attention_dim = 1, visualize_attention = True )

```
