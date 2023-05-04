Download Link: https://assignmentchef.com/product/solved-cs584-assignment-2-word-vectors
<br>
<h1></h1>

Homework assignments will be done individually: each student must hand in their own answers. Use of partial or entire solutions obtained from others or online is strictly prohibited. Electronic submission on Canvas is mandatory.

<ol>

 <li><strong>Basics </strong>(15 points)

  <ul>

   <li>(5pts)Prove that softmax is invariant to constant offset in the input, that is, for any input vector <strong>x </strong>and any constant c,</li>

  </ul></li>

</ol>

softmax(<strong>x</strong>) = softmax(<strong>x </strong>+ <em>c</em>)

where <strong>x </strong>+ <em>c </em>means adding the constant c to every dimension of <strong>x</strong>. Remember that

softmax(<strong>x</strong>)

<em>Note: In practice, we make use of this property and choose c </em>= −max<em><sub>i </sub>x<sub>i </sub>when computing softmax probabilities for numerical stability (i.e., subtracting its maximum element from all elements of x).</em>

<ul>

 <li>(5pts)Given an input matrix of N rows and D columns, compute the softmax prediction for each row using the optimization in part (a). Write your implementation in py. You may test by executing softmax.py</li>

 <li>(5pts)Derive the gradients of the sigmoid function and show that it can be rewritten as a function of the function value (i.e., in some expression where only <em>σ</em>(<em>x</em>), but not x, is present). Assume that the input x is a scalar for this question. Recall, the sigmoid function is:</li>

</ul>

Implement the sigmoid function in sigmoid.py and test your code.

<ol start="2">

 <li><strong>Word2vec </strong>(85 points)

  <ul>

   <li>(5pts) Assume you are given a predicted word vector <strong>v</strong><em><sub>c </sub></em>corresponding to the center word <em>c </em>for skipgram, and the word prediction is made with the softmax function</li>

  </ul></li>

</ol>

where <em>o </em>is the expected word, <em>w </em>denotes the <em>w</em>-th word and <strong>u</strong><em><sub>w </sub></em>(w = 1, …, W) are the “output” word vectors for all words in the vocabulary. The cross entropy function is defined as:

<em>J</em><sub>CE</sub>(<em>o,</em><strong>v</strong><em><sub>c</sub>,U</em>) = <em>CE</em>(<strong>y</strong><em>,</em><strong>y</strong>ˆ) = −<sup>X</sup><em>y<sub>i </sub></em>log(ˆ<em>y<sub>i</sub></em>)

<em>i</em>

where the gold vector <strong>y </strong>is a one-hot vector, the softmax prediction vector <strong>y</strong>ˆ is a probability distribution over the output space, and <em>U </em>= [<em>u</em><sub>1</sub><em>,u</em><sub>2</sub><em>,…,u<sub>W</sub></em>] is the matrix of all the output vectors. Assume cross entropy cost is applied to this prediction, derive the gradients with respect to <strong>v</strong><em><sub>c</sub></em>.

<ul>

 <li>(5pts)As in the previous part, derive gradients for the “output” word vector <strong>u</strong><em><sub>w </sub></em>(including <strong>u</strong><em><sub>o</sub></em>).</li>

 <li>(10pts)Repeat a and b assuming we are using the negative sampling loss for the predicted vector <strong>v</strong><em><sub>c</sub></em>. Assume that K negative samples (words) are drawn and they are 1,…,K respectively. For simplicity of notation, assume (<em>o /</em>∈ {1<em>,…,K</em>}). Again for a given word <em>o</em>, use <strong>u</strong><em><sub>o </sub></em>to denote its output vector. The negative sampling loss function in this case is:</li>

</ul>

<em>K</em>

<em>J</em>neg-sample(<em>o,</em><strong>v</strong><em>c,U</em>) = −log(<em>σ</em>(<strong>u</strong>&gt;<em>o </em><strong>v</strong><em>c</em>)) − Xlog(<em>σ</em>(−<strong>u</strong>&gt;<em>k </em><strong>v</strong><em>c</em>))

<em>k</em>=1

<ul>

 <li>(5pts)Derive gradients for all of the word vectors for skip-gram given the previous parts and given a set of context words [word<em><sub>c</sub></em><sub>−<em>m</em></sub><em>,…,</em>word<em><sub>c</sub>,…,</em>word<em><sub>c</sub></em><sub>+<em>m</em></sub>] where <em>m </em>is the context size. Denote the “input” and “output” word vectors for word <em>k </em>as <strong>v</strong><em><sub>k </sub></em>and <strong>u</strong><em><sub>k </sub></em></li>

</ul>

<em>Hint: feel free to use F</em>(<em>o,</em><strong>v</strong><em><sub>c</sub></em>) <em>(where o is the expected word) as a placeholder for the J<sub>CE</sub></em>(<em>o,</em><strong>v</strong><em><sub>c</sub>…</em>) <em>or J<sub>neg-sample</sub></em>(<em>o,</em><strong>v</strong><em><sub>c</sub>…</em>) <em>cost functions in this part – you’ll see that this is a useful abstraction for the coding part. That is, your solution may contain terms of the form </em><em> Recall that for skip-gram, the cost for a context centered around c is:</em>

X

<em>F</em>(<em>w</em><em>c</em>+<em>j</em><em>,</em><strong>v</strong><em>c</em>)

−<em>m</em>≤<em>j</em>≤<em>m,j</em>6=0

<ul>

 <li>(15pts) In this part you will implement the word2vec models and train your own word vectors with stochastic gradient descent (SGD). First, write a helper function to normalize rows of a matrix in py. In the same file, fill in the implementation for the softmax and negative sampling cost and gradient functions. Then, fill in the implementation of the cost and gradient functions for the skip-gram model. When you are done, test your implementation by running python word2vec.py.</li>

 <li>(15pts) Complete the implementation for your SGD optimizer in py. Test your implementation by running python sgd.py.</li>

 <li>(15pts) In this part you will implement the k-nearest neighbors algorithm, which will be used for analysis. The algorithm receives a vector, a matrix and an integer <em>k</em>, and returns <em>k </em>indices of the matrix’s rows that are closest to the vector. Use the cosine similarity as a distance metric <a href="https://en.wikipedia.org/wiki/Cosine_similarity">(</a><a href="https://en.wikipedia.org/wiki/Cosine_similarity">https://en.wikipedia.org/wiki/Cosine_similarity</a><a href="https://en.wikipedia.org/wiki/Cosine_similarity">)</a>. Implement the algorithm in py.</li>

 <li>(15pts)Show time! Now we are going to load some real data and train word vectors with everything you just implemented!</li>

</ul>

We are going to use the <a href="https://github.com/chtran/word2vec/tree/master/cs224d/datasets/stanfordSentimentTreebank">StanfordSentimentTreeBank</a> data to train word vectors. You need to process the dataset and use the sgd function and word2vec to generate word vectors. Visualize a few word examples. There is no additional code to write for this part; just run python run.py.