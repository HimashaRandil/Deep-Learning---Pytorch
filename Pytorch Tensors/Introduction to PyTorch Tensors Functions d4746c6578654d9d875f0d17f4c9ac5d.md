# Introduction to PyTorch Tensors Functions

In PyTorch, tensors are the fundamental data structures used to represent and manipulate multi-dimensional arrays. A tensor can be thought of as a generalized matrix with an arbitrary number of dimensions, including scalars, vectors, and matrices. Tensors are the basic building blocks for constructing neural network models and performing various mathematical operations.

### Key Characteristics:

1. **Data Types:** Tensors can store data of different types, such as integers, floats, or even more complex data structures.
2. **Shape:** The shape of a tensor defines its dimensions. For example, a 1D tensor represents a vector, a 2D tensor represents a matrix, and a tensor with more than two dimensions represents a multi-dimensional array.
3. **Operations:** PyTorch provides a wide range of operations for manipulating tensors, including arithmetic operations, matrix multiplication, element-wise operations, and more. These operations are crucial for building and training deep learning models.
4. **Automatic Differentiation:** Tensors in PyTorch are equipped with automatic differentiation capabilities through the Autograd module. This allows the tracking of operations on tensors to automatically compute gradients, facilitating gradient-based optimization algorithms.
5. **Interoperability:** PyTorch tensors seamlessly integrate with NumPy arrays, making it easy to transition between the two libraries. This interoperability is beneficial for data manipulation and integration with other scientific computing tools.

## Creating a Tensor

The easiest way to create a tensor is `torch.empty()` function:

```python
x = torch.empty(3,4)
print(type(x))
print(x)
```

![Untitled](Introduction%20to%20PyTorch%20Tensors%20Functions%20d4746c6578654d9d875f0d17f4c9ac5d/Untitled.png)

creates a 3x4 tensor named **`x`** with uninitialized values. The **`torch.empty`** function initializes a tensor with random or uninitialized values depending on the current state of the memory. It does not guarantee the values to be zero or any specific default values.

- 1-dimentional tensor is often called vector
- 2-dimentional tensor is often referred as matrix
- anything more than 2-dinmetion is referred to as tensor

 

```python
zeros = torch.zeros(2,3)
print(zeros)

ones = torch.ones(4,6)
print(ones)

torch.manual_seed(34514)
random = torch.rand(5,3)
print(random)
```

![Untitled](Introduction%20to%20PyTorch%20Tensors%20Functions%20d4746c6578654d9d875f0d17f4c9ac5d/Untitled%201.png)

**`zeros = torch.zeros(2,3)`**: Creates a 2x3 tensor named **`zeros`** filled with zeros.

**`ones = torch.ones(4,6)`**: Creates a 4x6 tensor named **`ones`** filled with ones.

**`torch.manual_seed(34514)`**: Sets the random seed to ensure reproducibility in generating random numbers.

**`random = torch.rand(5,3)`**: Creates a 5x3 tensor named **`random`** filled with random values between 0 and 1.

### Tensor Shapes

When performing operations with two or more tensors they need to be in same shape 

Which means they need to have same dimensions. That can be achieved using `torch.*like()` function

**`torch.empty_like(input)`**:

- **Purpose:** Creates a new tensor with the same shape as the input tensor but uninitialized (values are not initialized or set).

**`torch.ones_like(input)`**:

- **Purpose:** Creates a new tensor with the same shape as the input tensor, with all elements initialized to 1.

**`torch.zeros_like(input)`**:

- **Purpose:** Creates a new tensor with the same shape as the input tensor, with all elements initialized to 0.

**`torch.rand_like(input)`**:

- **Purpose:** Creates a new tensor with the same shape as the input tensor, with elements initialized to random values between 0 and 1.

```python
x = torch.empty(1, 3, 3)
print(x.shape)
print(x)

empty_like_x = torch.empty_like(x)
print(empty_like_x.shape)
print(empty_like_x)

ones_like_x = torch.ones_like(x)
print(ones_like_x.shape)
print(ones_like_x)

zeros_like_x = torch.zeros_like(x)
print(zeros_like_x.shape)
print(zeros_like_x)

rand_like_x = torch.rand_like(x)
print(rand_like_x.shape)
print(rand_like_x)
```

![Untitled](Introduction%20to%20PyTorch%20Tensors%20Functions%20d4746c6578654d9d875f0d17f4c9ac5d/Untitled%202.png)

### Creating using tuple or list

Using `torch.tensor()` is the most straightforward way to create a tensor if you already have data in a Python tuple or list. As shown above, nesting the collections will result in a multi-dimensional tensor.

```python
some_constants = torch.tensor([[6.2246, 7.2543], [9.52131,6.12421]])
print(type(some_constants))
print(some_constants)

some_integers = torch.tensor((5,7,2,55,98,3,2,75,32,6,2,9,21,76,0))
print(type(some_integers))
print(some_integers)

more_integers = torch.tensor(((2, 4, 6), [3, 6, 9]))
print(type(more_integers))
print(more_integers)
```

![Untitled](Introduction%20to%20PyTorch%20Tensors%20Functions%20d4746c6578654d9d875f0d17f4c9ac5d/Untitled%203.png)

## Tensor Data Types

In PyTorch, tensors can have different data types, each representing a specific kind of numerical data. The choice of data type affects the precision and memory usage of the tensor.

```python
a = torch.ones((1, 3), dtype=torch.int16)
print(a)

b = torch.rand((4, 4), dtype=torch.float64) * 20.
print(b)

c = b.to(torch.int32)
print(c)
```

![Untitled](Introduction%20to%20PyTorch%20Tensors%20Functions%20d4746c6578654d9d875f0d17f4c9ac5d/Untitled%204.png)

1. **`torch.float32` or `torch.float`**:
    - 32-bit floating-point number (default for most operations).

1. **`torch.float64` or `torch.double`**:
    - 64-bit floating-point number, providing higher precision.
    
2. **`torch.int32` or `torch.int`**:
    - 32-bit integer.
    
3. **`torch.int64` or `torch.long`**:
    - 64-bit integer.
    
4. **`torch.uint8`**:
    - 8-bit unsigned integer (commonly used for boolean values).
    
5. **`torch.bool`**:
    - Boolean data type (for representing True or False values).
    
6. **`torch.half`**:
    - 16-bit floating-point number (half-precision).
    
7. **Automatic Data Type Inference**:
    - PyTorch can automatically infer data types based on the input data.

```python
d = torch.ones((1, 3), dtype=torch.bool)
print(d)

e = torch.rand((4, 4), dtype=torch.double)
print(e)
```

![Untitled](Introduction%20to%20PyTorch%20Tensors%20Functions%20d4746c6578654d9d875f0d17f4c9ac5d/Untitled%205.png)

| Data type | dtype | CPU tensor | GPU tensor |
| --- | --- | --- | --- |
| 32-bit floating point | torch.float32 or torch.float | torch.FloatTensor | torch.cuda.FloatTensor |
| 64-bit floating point | torch.float64 or torch.double | torch.DoubleTensor | torch.cuda.DoubleTensor |
| 16-bit floating point https://pytorch.org/docs/stable/tensors.html#id4 | torch.float16 or torch.half | torch.HalfTensor | torch.cuda.HalfTensor |
| 16-bit floating point https://pytorch.org/docs/stable/tensors.html#id5 | torch.bfloat16 | torch.BFloat16Tensor | torch.cuda.BFloat16Tensor |
| 32-bit complex | torch.complex32 or torch.chalf |  |  |
| 64-bit complex | torch.complex64 or torch.cfloat |  |  |
| 128-bit complex | torch.complex128 or torch.cdouble |  |  |
| 8-bit integer (unsigned) | torch.uint8 | torch.ByteTensor | torch.cuda.ByteTensor |
| 8-bit integer (signed) | torch.int8 | torch.CharTensor | torch.cuda.CharTensor |
| 16-bit integer (signed) | torch.int16 or torch.short | torch.ShortTensor | torch.cuda.ShortTensor |
| 32-bit integer (signed) | torch.int32 or torch.int | torch.IntTensor | torch.cuda.IntTensor |
| 64-bit integer (signed) | torch.int64 or torch.long | torch.LongTensor | torch.cuda.LongTensor |
| Boolean | torch.bool | torch.BoolTensor | torch.cuda.BoolTensor |
| quantized 8-bit integer (unsigned) | torch.quint8 | torch.ByteTensor | / |
| quantized 8-bit integer (signed) | torch.qint8 | torch.CharTensor | / |
| quantized 32-bit integer (signed) | torch.qint32 | torch.IntTensor | / |
| quantized 4-bit integer (unsigned) https://pytorch.org/docs/stable/tensors.html#id6 | torch.quint4x2 | torch.ByteTensor | / |

---

## **Tensor class reference and functions**

**`Tensor.is_cuda()`**

- **Purpose:**
    - This method is used to determine whether a PyTorch tensor is stored on a GPU (CUDA device) or not.
- **Return Value:**
    - Returns a boolean value (**`True`** or **`False`**) indicating whether the tensor is located on a CUDA device (**`True`**) or not (**`False`**).
- **Usage:**

```python
# Create a tensor on CPU
cpu_tensor = torch.tensor([1, 2, 3])

# Check if the tensor is on CUDA (GPU)
is_cuda = cpu_tensor.is_cuda
print(is_cuda)  # Output: False
```

![Untitled](Introduction%20to%20PyTorch%20Tensors%20Functions%20d4746c6578654d9d875f0d17f4c9ac5d/Untitled%206.png)

```python
# Create a tensor on GPU
gpu_tensor = torch.tensor([1, 2, 3]).cuda()

# Check if the tensor is on CUDA (GPU)
is_cuda = gpu_tensor.is_cuda()
print(is_cuda)  # Output: True
```

**`Tensor.device`**

- **Purpose:**
    - This attribute is used to retrieve the device information on which a PyTorch tensor is stored.
- **Return Value:**
    - Returns a **`torch.device`** object that represents the device on which the tensor is stored.
- **Usage:**

```python

# Create a tensor on CPU
cpu_tensor = torch.tensor([1, 2, 3])
print(cpu_tensor.device)  # Output: cpu

```

![Untitled](Introduction%20to%20PyTorch%20Tensors%20Functions%20d4746c6578654d9d875f0d17f4c9ac5d/Untitled%207.png)

`Tensor.T`

- **Purpose:**
    - Transposes the given tensor
- **Return Value:**
    - Returns a new tensor that is the transposed version of the original tensor.
- **Usage:**

```python
# Create a 2D tensor
original_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Transpose the tensor
transposed_tensor = original_tensor.T

print("Original Tensor:")
print(original_tensor)
print("Transposed Tensor:")
print(transposed_tensor)
```

![Untitled](Introduction%20to%20PyTorch%20Tensors%20Functions%20d4746c6578654d9d875f0d17f4c9ac5d/Untitled%208.png)

**`Tensor.H`**

- **Purpose:**
    - Use to transpose and conjugate a tensor
- **Return Value:**
    - Returns a new tensor that is the transposed and conjugated version of the original tensor.
- **Usage:**

```python
# Create a 2D tensor
original_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Transpose the tensor
transposed_and_conjugated_tensor = original_tensor.H

print("Original Tensor:")
print(original_tensor)
print("Transposed and Conjugated Tensor:")
print(transposed_and_conjugated_tensor)
```

![Untitled](Introduction%20to%20PyTorch%20Tensors%20Functions%20d4746c6578654d9d875f0d17f4c9ac5d/Untitled%209.png)

**`Tensor.is_meta`**

- **Purpose:**
    
    method that checks whether a tensor is a meta tensor. Meta tensors are a special type of tensor that represent the **symbolic information** (shape, dtype, device) of a tensor, but they don't hold any actual data. They are often used in certain operations, such as:
    
    - **Automatic differentiation (autograd):** During autograd, PyTorch creates meta tensors to track the computational graph, which is a sequence of operations that generate the final result. These meta tensors act as placeholders for the actual tensors that will be created during the forward pass.
    - **Lazy tensors:** These are tensors that defer the allocation of memory and computation until they are absolutely necessary. Meta tensors can be used to represent lazy tensors, as they only hold the symbolic information until the actual data is needed.
- **Return Value:**
    - **Is `True` if the Tensor is a meta tensor, `False` otherwise.**
- **Usage:**

```python
# Create a regular tensor
x = torch.tensor([1, 2, 3])

# Check if it's a meta tensor (should be False)
print(x.is_meta)  # Output: False

```

![Untitled](Introduction%20to%20PyTorch%20Tensors%20Functions%20d4746c6578654d9d875f0d17f4c9ac5d/Untitled%2010.png)

**`Tensor.grad`**

- **Purpose:**
    - An attribute that holds the **gradient** of a tensor with respect to another tensor
- **Usage:**
    - During the training of neural networks, you perform forward and backward passes. The **`grad`** attribute is populated during the backward pass and provides access to the gradient of the tensor.
- **Example:**

```python
# Create a tensor and perform some operations
x = torch.tensor([2.0], requires_grad=True)
y = x**2 + 3*x + 1

# Perform backward pass to compute gradients
y.backward()

# Access the gradient of the tensor with respect to the scalar value
gradient = x.grad # after differenciation 2x+3 = 2(2.0)+3 = 7
print(gradient)
```

![Untitled](Introduction%20to%20PyTorch%20Tensors%20Functions%20d4746c6578654d9d875f0d17f4c9ac5d/Untitled%2011.png)

**`tensor.clone()`**

- **Purpose:**
    - Creates a deep copy of the tensor, i.e., a new tensor with the same data but stored in a different memory location. (Different Variable)
- **Return Value:**
    - Returns a new tensor that is a copy of the original tensor.
- **Usage:**

```python
# Create a tensor
original_tensor = torch.tensor([1, 2, 3])

# Clone the tensor
cloned_tensor = original_tensor.clone()
print(cloned_tensor)
```

![Untitled](Introduction%20to%20PyTorch%20Tensors%20Functions%20d4746c6578654d9d875f0d17f4c9ac5d/Untitled%2012.png)

**`tensor.sum(dim=None, keepdim=False)`**

- **Purpose:**
    - Computes the sum of the elements along a specified dimension.
- **Usage:**

```python
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Sum along dimension 0 (columns)
sum_along_dim0 = tensor.sum(dim=0)
print(sum_along_dim0)

# Sum along dimension 1 (rows)
sum_along_dim1 = tensor.sum(dim=1)
print(sum_along_dim1)
```

![Untitled](Introduction%20to%20PyTorch%20Tensors%20Functions%20d4746c6578654d9d875f0d17f4c9ac5d/Untitled%2013.png)

All bellow fuctions work as same as the above 

**`tensor.prod(dim=None, keepdim=False)`**

- **Purpose:**
    - Computes the product of the elements along a specified dimension.

**`tensor.mean(dim=None, keepdim=False)`**

- **Purpose:**
    - Computes the mean of the elements along a specified dimension.

**`tensor.min(dim=None, keepdim=False)`**

- **Purpose:**
    - Returns the minimum value and its index along a specified dimension.

**`tensor.max(dim=None, keepdim=False)`**

- **Purpose:**
    - Returns the maximum value and its index along a specified dimension.
    

`Tensor.corrcoef`

- **Purpose:**
    - A function that calculates the **Pearson product-moment correlation coefficient matrix** between the variables in a given tensor. It's used to measure the **linear relationship** between two or more variables.
    - The Pearson correlation coefficient (PCC) ranges from -1 to 1:
        - **1**: Perfect positive correlation (variables increase or decrease together).
        - **0**: No linear correlation.
        - **1**: Perfect negative correlation (as one variable increases, the other decreases).
- **Usage:**

```python
# Sample data
data = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Calculate correlation coefficients (between rows)
corr_matrix = torch.corrcoef(data)
print(corr_matrix)
```

![Untitled](Introduction%20to%20PyTorch%20Tensors%20Functions%20d4746c6578654d9d875f0d17f4c9ac5d/Untitled%2014.png)

`Tensor.count_nonzero() / torch.count_nonzero()`

- **Purpose:**
    - `Tensor.count_nonzero` (or `tf.math.count_nonzero` for TensorFlow) is a function that counts the number of **non-zero elements** in a tensor. It can be a helpful tool for analyzing the sparsity of a tensor or identifying the number of active elements after applying activation functions.
- **Usage:**

```python
# Sample tensor
tensor = torch.tensor([[1, 0, 3], [0, 4, 0], [0,0,0], [0,3,5], [4,8,2]])

# Count non-zero elements across all dimensions
non_zero_count = tensor.count_nonzero()
print(non_zero_count)  

# Count non-zero elements along the first dimension (rows)
non_zero_count_rows = tensor.count_nonzero(dim=0)
print(non_zero_count_rows)
```

![Untitled](Introduction%20to%20PyTorch%20Tensors%20Functions%20d4746c6578654d9d875f0d17f4c9ac5d/Untitled%2015.png)

**`torch.diag`**

- **Purpose:**
    - A function that takes a diagonal vector or a matrix and **extracts its diagonal elements** into a new 1D tensor. It's useful for working with diagonal components separately.
- **Usage:**

```python
# Sample matrix
matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Extract diagonal elements
diagonal = torch.diag(matrix)
print(diagonal)  
```

![Untitled](Introduction%20to%20PyTorch%20Tensors%20Functions%20d4746c6578654d9d875f0d17f4c9ac5d/Untitled%2016.png)

`tensor.dot`

- **Purpose:**
    - A function that performs a standard **matrix multiplication**  on two tensors.
    - (Currently the function is having an error)
- **Usage:**

```python

# Sample tensors
matrix2 = torch.tensor([[1, 2], [3, 4]])
matrix1 = torch.tensor([[5, 6], [7, 8]])

# Matrix multiplication
result = torch.dot(matrix1,matrix2)
print(result)
```

`tensor.reshape`

- **Purpose:**
    - A function that allows you to change the shape (dimensions) of a tensor to a new desired shape, as long as the total number of elements remains the same.
- **Usage:**

```python
# Sample tensor
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Reshape to a row vector (view)
reshaped_tensor = tensor.reshape(-1)  # Equivalent to tensor.reshape(6)
print(reshaped_tensor)  # Output: tensor([1, 2, 3, 4, 5, 6])
```

![Untitled](Introduction%20to%20PyTorch%20Tensors%20Functions%20d4746c6578654d9d875f0d17f4c9ac5d/Untitled%2017.png)

`tensor.unique`

- **Purpose:**
    - A function that allows you to change the shape (dimensions) of a tensor to a new desired shape, as long as the total number of elements remains the same.
- **Usage:**

```python
tensor = torch.tensor([2, 1, 2, 3, 1, 4])

# Get unique elements (sorted)
unique_elements = tensor.unique()
print(unique_elements)
```

![Untitled](Introduction%20to%20PyTorch%20Tensors%20Functions%20d4746c6578654d9d875f0d17f4c9ac5d/Untitled%2018.png)