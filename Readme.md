# Description
With my teammate [@Ghali](https://github.com/Ghali9)
for a computer vision course , we have to build a simple vision project. 

This project must use open CV and an IA .

We choose python to code this project and use TensorFlow to quickly integrate  an IA.


# steps 

1) we catch a picture of our chessborad and apply a HSV filter.

![HSV](./img/HSV.png)

2) with the HSV picture we determine the mask to find the corners of the chessboard.

![MASK](./img/mask.png)

3) when we have corners we can use opencv to build a square picture of our chessboard.

![BOARD](./img/board.jpg)

4) after that we simply cut the pictures into 64 cases 

![cases](./img/cases.png)

5) After prosessing a lot of picture we sort them into the dataset file .

![dataset](./img/dataset.png)

6) we create a model 

7) we use the model to determine each case value . 

![result1](./img/result1.png)

8) All picture are not working with a big accuracy ...

![result2](./img/result2.png)

