/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuralnetwork;

import com.mycompany.neuralnetwork.exceptions.MatrixException;
import com.mycompany.neuralnetwork.matrix.Matrix;
import com.mycompany.neuralnetwork.matrix.Matrices;
import com.mycompany.neuralnetwork.network.Network;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Arrays;
import java.util.List;
import com.google.gson.Gson;
import com.mycompany.neuralnetwork.cost.CrossEntropyCost;
import com.mycompany.neuralnetwork.dataloader.DataLoader;
import com.mycompany.neuralnetwork.layers.ConvLayer;
import com.mycompany.neuralnetwork.layers.FlatLayer;
import com.mycompany.neuralnetwork.layers.InputLayer;
import com.mycompany.neuralnetwork.layers.OutputLayer;
import com.mycompany.neuralnetwork.layers.PoolLayer;
import com.mycompany.neuralnetwork.network.MultiLayerNetwork;
import com.mycompany.neuralnetwork.network.NetworkEnhanced;
import com.mycompany.neuralnetwork.network.NetworkNd4j;
import com.mycompany.neuralnetwork.neuron.MaxNeuron;
import com.mycompany.neuralnetwork.neuron.SigmoidNeuron;
import com.mycompany.neuralnetwork.neuron.TanhNeuron;
import com.mycompany.neuralnetwork.optimizers.Adagrad;
import com.mycompany.neuralnetwork.optimizers.RMSprop;
import com.mycompany.neuralnetwork.optimizers.SGD;
import java.util.ArrayList;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
//import org.deeplearning4j.nn.conf.layers.OutputLayer;
//import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;
/**
 *
 * @author dmytr
 */
public class NeuralNetwork {
    
    public static void main(String[] args) throws MatrixException, Exception{
        Matrix matrix=new Matrix(new int[]{9},Integer.class);
        Matrix newMatrix=new Matrix(new int[]{9});
        //List<Number> matrixElems=Arrays.asList(new Number[]{1,2,3,4,5,6,7,8,9,10,11,12});
        List<Number> matrixElems=Arrays.asList(new Number[]{1,2,3,4,5,6,7,8,7});
        matrix.setElements(matrixElems);        
        List<Number> newMatrixElems=Arrays.asList(new Number[]{1,2,3,4,5,6,7,8,9});
        newMatrix.setElements(newMatrixElems);
        System.out.println("Matrix:"+matrix.getElements().size());
        System.out.println(matrix.toString());
        System.out.println("newMatrix:");
        System.out.println(newMatrix.toString());
        //matrix.add(newMatrix);
        //matrix.applyFunc((e)->e=e.doubleValue()+2);
        //matrix.transpose();
        //matrix.add(1d);
        //matrix.matrixMult(newMatrix);
        //System.out.println("Matrixes addition:"+Matrixes.add(matrix,newMatrix));
        System.out.println("Result:");
        System.out.println(matrix.toString());
        //System.out.println(Matrices.applyFunc(matrix, (i)->i*2));
        //System.out.println(matrix.sum(0).toString());
        //System.out.println("Combine lists result:");
        //System.out.println(Matrixes.combineElements(Arrays.asList(new Integer[]{1,2,3,4,5}),Arrays.asList(new Integer[]{11,12,30,24,50})));
        //Matrix input=new Matrix(Arrays.asList(new Integer[]{1,1,1}),new int[]{3},Double.class);
        //Matrix output=new Matrix(Arrays.asList(new Integer[]{1,1}),new int[]{2},Double.class);
        INDArray matrix100=Nd4j.create(IntStream.iterate(1, (i)->i+1).limit(100).toArray(),new long[]{100}, DataType.INT32);        
        INDArray pseudo_weights=Nd4j.create(IntStream.generate(()->2).limit(50).toArray(),new long[]{2,1,1,5,5},DataType.INT32);
        INDArray pseudo_deltas=Nd4j.create(IntStream.generate(()->2).limit(72).toArray(),new long[]{2,6,6,1,1},DataType.INT32);        
        ConvLayer conv=new ConvLayer(new int[]{1,10,10},2,new int[]{5,5},new SigmoidNeuron());
        PoolLayer poolLayer=new PoolLayer(new int[]{1,10,10},2,new int[]{5,5},new SigmoidNeuron());
        conv.setWeights(pseudo_weights);
        //System.out.println("matrix100*pseudo_weights:"+matrix100.mul(pseudo_weights));
        System.out.println("Matrix100 parsed:\n"+conv.parseImage(matrix100,new int[]{1,10,10},new int[]{5,5},true));
        System.out.println("Matrix100 parsed pool:\n"+poolLayer.parseImageNew(matrix100));
        System.out.println("Matrix100 mulConv:\n"+conv.mulConv(matrix100));
        System.out.println("Deltas positions:\n"+Arrays.stream(conv.getDeltasPositions(new int[]{3,3},true))
                .map(arr->Arrays.toString(arr)).collect(Collectors.joining(",")));
        System.out.println("Matrix100*pseudo_weights:\n"+conv.parseImage(matrix100,new int[]{2,10,10},new int[]{5,5},true).mul(pseudo_weights).sum(3,4));
        System.out.println("Matrix100_sum(1,2)*2:\n"+conv.parseImage(matrix100,new int[]{2,10,10},new int[]{5,5},true).sum(1,2).mul(2));
        System.out.println("Matrix100*pseudo_deltas:\n"+conv.parseImage(matrix100,new int[]{2,10,10},new int[]{5,5},true).mul(pseudo_deltas).sum(1,2));
        INDArray twos=Nd4j.ones(3,2,1).mul(2);
        INDArray threes=Nd4j.ones(3,1,2).mul(3);
        System.out.println("Mmul result:"+Nd4j.matmul(twos, threes));
        INDArray pseudo_weights_short=Nd4j.create(IntStream.generate(()->2).limit(25).toArray(),new long[]{5,5},DataType.INT32);
        matrix100=matrix100.reshape(10,10);
        matrix100.put(new INDArrayIndex[]{NDArrayIndex.interval(0,5),
            NDArrayIndex.interval(0,5)},matrix100.get(NDArrayIndex.interval(0,5),
                    NDArrayIndex.interval(0,5)).mul(pseudo_weights_short));
        System.out.println("Matrix100 updated:"+matrix100);
        System.out.println("pseudo_weights get:"+pseudo_weights.get(NDArrayIndex.interval(0,1),
                NDArrayIndex.interval(0,1),
                NDArrayIndex.interval(0,1)));
        INDArray matrix10=Nd4j.create(IntStream.iterate(1, (i)->i+1).limit(10).toArray(),new long[]{10}, DataType.INT32);
        INDArray pool=Nd4j.create(new int[]{1,2,3,4},new long[]{2,2},DataType.INT32);
        INDArray delta=Nd4j.matmul(matrix10.reshape(new long[]{1,matrix10.shape()[0]/1,1})//test
                ,pool.reshape(1,1,pool.shape()[1]*pool.shape()[0]));
        System.out.println("Deltas:"+delta.reshape(10,2,2));
        //Nd4j.matmul(twos, threes);
        //System.out.println(conv.parseImage(matrix100,new int[]{5,5},new int[]{10,10},true).sum(2,3));
        //Network net=new Network(new int[]{784,30,10});
        //NetworkNd4j net=new NetworkNd4j(new int[]{784,30,10});
        //NetworkEnhanced net=new NetworkEnhanced(new int[]{784,30,10}, new CrossEntropyCost(), new SigmoidNeuron());
        MultiLayerNetwork net=new MultiLayerNetwork(new InputLayer(),
                new ConvLayer(new int[]{20,28,28},20,new int[]{5,5},new SigmoidNeuron()),
                new PoolLayer(new int[]{20,24,24},20,new int[]{2,2},new MaxNeuron()),
                new FlatLayer(2880,100,new SigmoidNeuron()),
                new OutputLayer(100,10,new SigmoidNeuron(),new CrossEntropyCost()));
        /*System.out.println("Feedforward:"+
                net.feedforward(new Matrix(Arrays.asList(new Integer[]{1,0,1}),new int[]{3},Double.class)));
        System.out.println("Sigmoid:"+net.sigmoid(10));
        System.out.println("Sisgmoid_prime:"+net.sigmoid_prime(10));
        System.out.println("Cost_derivative:"+net.cost_derivative(matrix,newMatrix));
        System.out.println("Evaluate:"+net.evaluate(Arrays.asList(new Matrix[][]{new Matrix[]{input,output}})));*/
        //System.out.println("Backpropagation"+Arrays.toString(net.backprop(input,output)[1]));
        //long memoryBefore=Runtime.getRuntime().totalMemory()-Runtime.getRuntime().freeMemory();
        //List<Matrix[]> training_data=DataLoader.loadTrainingDataAsMatrices(1);
        //List<Matrix[]> test_data=DataLoader.loadTestDataAsMatrices();
        System.out.println("CrossEntropyCost:"+new CrossEntropyCost().delta(Nd4j.ones(10), Nd4j.zeros(10), null));
        System.out.println("Max:"+Transforms.max(Nd4j.create(new int[]{1,-1,-2,3,4,-1,5,0,7},new long[]{3,3},DataType.INT32),0));
        System.out.println("Max derivative:"+new MaxNeuron().derivative(Nd4j.create(new int[]{1,-1,-2,3,4,-1,5,0,7},new long[]{3,3},DataType.INT32)));
        List<INDArray[]> training_data_nd4j=DataLoader.loadTrainingDataAsINDArrays(1);
        List<INDArray[]> test_data_nd4j=DataLoader.loadTestDataAsINDArrays();
        //net.setWeights(Arrays.asList(new INDArray[]{Nd4j.ones(new int[]{1}).castTo(DataType.DOUBLE)}));
        //net.setBiases(Arrays.asList(new INDArray[]{Nd4j.ones(new int[]{1}).castTo(DataType.DOUBLE)}));
        /*List<INDArray[]> training_data_nd4j=IntStream.range(0, 1000).boxed().map(
                i->new INDArray[]{Nd4j.ones(new long[]{1}).castTo(DataType.DOUBLE),
                    Nd4j.zeros(new long[]{1}).castTo(DataType.DOUBLE)}).collect(Collectors.toList());
        List<INDArray[]> test_data_nd4j=IntStream.range(0, 1000).boxed().map(
                i->new INDArray[]{Nd4j.ones(new long[]{1}).castTo(DataType.DOUBLE),
                    Nd4j.zeros(new long[]{1}).castTo(DataType.DOUBLE)}).collect(Collectors.toList());*/
        //System.out.println(test_data_nd4j.get(0)[1]);
        //System.out.println(net.vectorized_result(test_data_nd4j.get(0)[1]));
        //test_data.stream().limit(100).forEach(m->System.out.println("Elements"+m[1].getElements()));
        /*matrix.getElements().stream().mapToInt(e->e.intValue()).toArray();
        INDArray nd_matrix=Nd4j.create(matrix.getElements().stream().mapToInt(e->e.intValue()).toArray(),
                Arrays.stream(matrix.getShape()).mapToLong(i->(long)i).toArray(),DataType.DOUBLE);
        INDArray nd_matrix1=Nd4j.create(newMatrix.getElements().stream().mapToInt(e->e.intValue()).toArray(),
                Arrays.stream(newMatrix.getShape()).mapToLong(i->(long)i).toArray(),DataType.DOUBLE);
        System.out.println(Arrays.toString(nd_matrix.shape()));
        System.out.println("Result of operation:"+nd_matrix.sub(nd_matrix1).toString());
        System.out.println("nd_matrix:"+nd_matrix.argMax(0).data().asInt()[0]);
        System.out.println(Arrays.toString(Matrices.matrixMultNd4j(matrix, newMatrix).getShape()));
        System.out.println(matrix);
        System.out.println(matrix.argmax());*/
        //System.out.println(test_data.get(0)[1].getElements().get(0).intValue());
        //System.out.println(matrix.argmax()==test_data.get(0)[1].getElements().get(0).intValue());
        /*System.out.println("Training_data memory:"+(memoryBefore-(Runtime.getRuntime().totalMemory()-Runtime.getRuntime().freeMemory())));
        System.out.println("Training_data size:"+training_data.size());
        System.out.println("Shape:"+Arrays.toString(training_data.get(0)[0].getShape()));
        System.out.println("Elements:"+training_data.get(0)[0].getElements());
        System.out.println("ElType:"+training_data.get(0)[0].getElType());
        System.out.println("Actual elType:"+training_data.get(0)[0].getElements().get(0).doubleValue()+2);
        System.out.println("Size:"+training_data.get(0)[0].getSize());
        System.out.println(training_data.get(0)[1].getElements());
        System.out.println("Argmax:"+training_data.get(0)[1].argmax());
        System.out.println("Argmax matrix:"+matrix.argmax());
        System.out.println("max matrix:"+matrix.getElements().stream().map(e->e.doubleValue()).max(Double::compare).get());
        System.out.println("Argmax matrix:"+matrix.getElements().stream().map(e->e.doubleValue()).collect(Collectors.toList()).indexOf(matrix.getElements().stream().map(e->e.doubleValue()).max(Double::compare).get()));
        System.out.println("Matrix elType:"+matrix.getElements().get(0).getClass());*/
        //net.backprop(training_data.get(0)[0], training_data.get(0)[1]);
        //net.setWeights(Arrays.asList(new Matrix[]{new Matrix(IntStream.generate(()->1).limit(784*30).boxed().map(i->i.floatValue()*0.1).collect(Collectors.toList()),new int[]{30,784},Float.class),new Matrix(IntStream.generate(()->1).limit(300).boxed().map(i->i.floatValue()*0.01).collect(Collectors.toList()),new int[]{10,30},Float.class)}));
        //net.setBiases(Arrays.asList(new Matrix[]{new Matrix(IntStream.generate(()->1).limit(30).boxed().map(i->i.floatValue()*0.01).collect(Collectors.toList()),new int[]{30},Float.class),new Matrix(IntStream.generate(()->1).limit(10).boxed().map(i->i.floatValue()*0.1).collect(Collectors.toList()),new int[]{10},Float.class)}));
        //net.setWeights(Arrays.asList(Nd4j.ones(DataType.DOUBLE,30,784),Nd4j.ones(DataType.DOUBLE,10,30)));
        //net.setBiases(Arrays.asList(Nd4j.ones(DataType.DOUBLE,30),Nd4j.ones(DataType.DOUBLE,10)));
        //System.out.println(net.getBiases());
        //System.out.println(net.evaluate(test_data.subList(0, 100)));
        //System.out.println(net.feedforward(training_data_nd4j.get(0)[0]));
        //net.SGD(training_data, 5, 10, 0.5d, test_data.subList(0, 1000));
        //net.SGD(training_data_nd4j, 10, 10, 0.5d, test_data_nd4j);
        //net.SGD(training_data_nd4j, 10, 10, 0.5d, 5d, false, false, false, test_data_nd4j);
        //new Adagrad().optimize(net, training_data_nd4j, 10, 10, 0.00005d, 5d, false, false, false, test_data_nd4j);
        //new RMSprop().optimize(net, training_data_nd4j.subList(0, 1000), 10, 10, 0.001d, 0.9d, false, false, false, test_data_nd4j.subList(0, 1000));
        System.out.println(training_data_nd4j.get(0)[0]);
        net.SGD(training_data_nd4j.subList(0, 100), 10, 10, 0.5d, 5d, test_data_nd4j.subList(0, 100));
        //System.out.println(training_data_nd4j.get(0)[0]);
        //System.out.println(Arrays.toString(Nd4j.create(new double[]{0.76498521,0.28495175,2.25854695},
        //        new long[]{3L},DataType.DOUBLE).mul(2.24175968).data().asDouble()));
        //net.update_mini_batch(training_data_nd4j.subList(0, 1), 0.5d, 5d, 1);
        //System.out.println(Arrays.toString(net.feedforward(training_data_nd4j.get(0)[0]).data().asDouble()));
        //System.out.println(Nd4j.ones(DataType.DOUBLE,9).mmul(Nd4j.ones(DataType.DOUBLE,9).add(1)));
        //System.out.println("Backprop:\n"+Arrays.toString(net.backprop(training_data_nd4j.get(0)[0], training_data_nd4j.get(0)[1])[1]));
        //System.out.println(Arrays.toString(net.backprop(Nd4j.ones(DataType.DOUBLE,784).mul(0.5), Nd4j.ones(DataType.DOUBLE,10))));
        //System.out.println(net.feedforward(training_data.get(0)[0]));
        //System.out.println(net.evaluate(test_data.subList(0, 1000)));
        //System.out.println(net.feedforward(training_data.get(0)[0]));
        /*FileReader fr=new FileReader("src\\main\\resources\\data\\mnist_training_data_by_line.json");
        BufferedReader br=new BufferedReader(fr);
        String s=null;
        StringBuilder text=new StringBuilder(br.readLine());
        long beginPoint=System.currentTimeMillis();
        System.out.println(text.substring(text.length()-100));*/
        //System.out.println(net.feedforward(training_data.get(0)[0]));
        /*Matrix training_input=new Matrix(new int[]{10},Double.class);
        Matrix training_output=new Matrix(Arrays.asList(new Double[]{0d,0d,0d,1d,0d}),new int[]{5},Double.class);
        System.out.println("I'm here");
        training_input.setElements(IntStream.generate(()->1).limit(10).boxed().map(i->i.doubleValue()).collect(Collectors.toList())); 
        net.setWeights(Arrays.asList(new Matrix[]{new Matrix(IntStream.generate(()->1).limit(30).boxed().map(i->i.doubleValue()).collect(Collectors.toList()),new int[]{3,10},Double.class),new Matrix(IntStream.generate(()->1).limit(15).boxed().map(i->i.doubleValue()).collect(Collectors.toList()),new int[]{5,3},Double.class)}));
        net.setBiases(Arrays.asList(new Matrix[]{new Matrix(IntStream.generate(()->1).limit(3).boxed().map(i->i.doubleValue()).collect(Collectors.toList()),new int[]{3},Double.class),new Matrix(IntStream.generate(()->1).limit(5).boxed().map(i->i.doubleValue()).collect(Collectors.toList()),new int[]{5},Double.class)}));
        */
        //System.out.println("Bias grad:"+Arrays.toString(net.backprop(training_input, training_output)[0]));
        //System.out.println("Weight grad:"+Arrays.toString(net.backprop(training_input, training_output)[1]));
        //test_data.stream().limit(100).forEach(m->System.out.println("Elements"+m[1].getElements()));
        /*net.update_mini_batch(Arrays.asList(new Matrix[][]{new Matrix[]{test_data.get(0)[1],training_output}}), 0.5);
        System.out.println("Biases:"+net.getBiases());
        System.out.println("Weights:"+net.getWeights());*/
        //Matrix backprop_input=new Matrix(DoubleStream.generate(()->0.5).limit(10).boxed().collect(Collectors.toList()),new int[]{10},Double.class);
        //System.out.println(Matrices.matrixMultNd4j(net.getWeights().get(0), backprop_input));
        //System.out.println(Arrays.toString(net.backprop(backprop_input, training_output)[0]));
        //List<Matrix[]> training_data=new ArrayList<>();
        //List<Matrix[]> test_data=new ArrayList<>();
        //Random rand=new Random();
        /*for(int i=0;i<10000;i++){
            training_input.setElements(Arrays.asList(new Double[]{rand.nextDouble(),rand.nextDouble(),rand.nextDouble()}));
            training_data.add(new Matrix[]{training_input,training_output});
            if(i<5000){
                training_input.setElements(Arrays.asList(new Double[]{rand.nextDouble(),rand.nextDouble(),rand.nextDouble()}));
                test_data.add(new Matrix[]{training_input,training_output});
            }
        }*/
        //System.out.println(net.feedforward(input));
        //net.SGD(training_data, 10, 50, 1, test_data);
        //System.out.println(net.feedforward(input));
        //int zeros=0;
        /*for(Matrix[] m:training_data){
            Matrix result=net.feedforward(m[0]);
            if(Math.round(result.getElements().get(1).doubleValue())==1 && Math.round(result.getElements().get(0).doubleValue())==0)zeros++;
        }*/
        //System.out.println(zeros);
        //Matrix[][] training_data=DataLoader.loadValidationData();
        //System.out.println(training_data[0][1]);
        //List<Matrix[]> training_data=DataLoader.loadTrainingData();
        //Matrix train_output=training_data.get(0)[1];
        /*System.out.println("Training data size:"+training_data.size());
        System.out.println("Training data datatype:"+(training_data.get(0))[0].getClass());
        System.out.println("Training data example:\n"+(training_data.get(0))[1]);*/
        /*System.out.println("Shape:"+Arrays.toString(train_output.getShape()));
        System.out.println("Size:"+train_output.getSize());
        System.out.println("Axes:"+train_output.getAxes());
        System.out.println(train_output);*/
        /*Matrix w=new Matrix(Arrays.asList(1,2,3,4,5,6,7,8,9),new int[]{3,3},Integer.class);
        Matrix a=new Matrix(Arrays.asList(1,2,3),new int[]{3},Integer.class);
        System.out.println(Matrices.transpose(w));
        System.out.println(Matrices.multiply(a,Matrices.transpose(w)).sum(0));*/
        /*List<Matrix> weights=new ArrayList<>();
        weights.add(new Matrix(IntStream.generate(()->1).limit(9).boxed().collect(Collectors.toList()),new int[]{3,3}, Double.class));
        weights.add(new Matrix(IntStream.generate(()->1).limit(6).boxed().collect(Collectors.toList()),new int[]{2,3}, Double.class));
        List<Matrix> biases=new ArrayList<>();
        biases.add(new Matrix(IntStream.generate(()->1).limit(3).boxed().collect(Collectors.toList()),new int[]{3}, Double.class));
        biases.add(new Matrix(IntStream.generate(()->1).limit(2).boxed().collect(Collectors.toList()),new int[]{2}, Double.class));
        net.setBiases(biases);
        net.setWeights(weights);
        System.out.println("Backpropagation check:"+net.backprop(input, output));*/
        /*DataSetIterator mnistTrain=new MnistDataSetIterator(128,true,123);
        DataSetIterator mnistTest=new MnistDataSetIterator(128,false,123);
        MultiLayerConfiguration config=new NeuralNetConfiguration.Builder().activation(Activation.SIGMOID)
                .weightInit(WeightInit.XAVIER).l2(0.0001).seed(123).list()
                .layer(0, new DenseLayer.Builder().nIn(784).nOut(30).build())
                .layer(1, new DenseLayer.Builder().nIn(30).nOut(10).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX).nIn(30).nOut(10).build())
                .build();*/
        //MultiLayerNetwork model=new MultiLayerNetwork(config);
        //model.init();
        //model.fit(mnistTest);
        //Evaluation eval=model.evaluate(mnistTest);
        //System.out.println(eval.stats());
        
    }
    
}
