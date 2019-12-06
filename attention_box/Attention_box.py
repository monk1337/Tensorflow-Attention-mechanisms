import tensorflow as tf

class Attention_box(object):

    @staticmethod
    def soft_attention(query, attention_size, visualize_attention = False):

        # query is three dim tensor
        # batch x max sequence length x dim { if output is from bi-lstm }
        # other output also should be three dim

        dim_shape      = query.shape[2]
        reshape_tensor = tf.reshape(query,[-1,dim_shape])


        attention_size = tf.get_variable(name='attention_size',
                                         shape=[dim_shape,attention_size],
                                         dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer(-0.01,0.01))
        # bias 1
        bias          = tf.get_variable(name='bias',shape=[attention_size],
                                        dtype=tf.float32,
                                        initializer=tf.random_uniform_initializer(-0.01,0.01))


        attention_projection = tf.add(tf.matmul(reshape_tensor,attention_size),bias)
        output_reshape       = tf.reshape(attention_projection,[tf.shape(query)[0],tf.shape(query)[1],-1])
        attention_output     = tf.nn.softmax(output_reshape,dim = 1)



        attention_visualize = tf.reshape(attention_output,
                                         [tf.shape(query)[0],
                                          tf.shape(query)[1]],
                                         name='Plot')
        
        
   
        attention_projection_output = tf.multiply(attention_output,query)
        

        Final_output = tf.reduce_sum(attention_projection_output,1)

        if visualize_attention:
            
            return {
                    'attention_output': attention_projection_output, 
                    'visualize vector': attention_visualize,
                    'reduced_output'   : Final_output
                    }

                
        