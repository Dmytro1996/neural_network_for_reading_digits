/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuralnetwork.network;

import com.mycompany.neuralnetwork.exceptions.MatrixException;
import com.mycompany.neuralnetwork.matrix.Matrices;
import com.mycompany.neuralnetwork.matrix.Matrix;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.function.Function;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 *
 * @author dmytr
 */
public class NetworkNd4j {
    
    private int num_layers;
    private int[] shape;
    private List<INDArray> weights;
    private List<INDArray> biases;
    
    public NetworkNd4j(int[] shape){
        this.shape=shape;
        this.num_layers=shape.length;
        weights=new LinkedList<>();
        biases=new LinkedList<>();
        Random rand=new Random();
        for(int i=0;i<num_layers;i++){
            if(i<num_layers-1)weights.add(Nd4j.create(DoubleStream.generate(()->rand.nextGaussian())
                    .limit(shape[i+1]*shape[i]).toArray(),
                    new long[]{shape[i+1],shape[i]},DataType.DOUBLE));
            if(i>0)
                biases.add(Nd4j.create(DoubleStream.generate(()->rand.nextGaussian())
                        .limit(shape[i]).toArray(),new long[]{shape[i]},DataType.DOUBLE));
        }
    }
    
    public INDArray feedforward(INDArray activation){
        for(int i=0;i<weights.size();i++){
            activation=Transforms.sigmoid(weights.get(i).mmul(activation).add(biases.get(i)),true);
        }
        return activation;
    }
    
    public void SGD(List<INDArray[]> train_data, int epochs, int mini_batch_size, double eta,List<INDArray[]>...test_data){
        int len_test=(test_data.length!=0)?test_data[0].size():0;
        int len_train=train_data.size();
        for(int i=0;i<epochs;i++){
            Collections.shuffle(train_data);
            List<List<INDArray[]>> mini_batches=IntStream.range(0,(len_train/mini_batch_size))
                    .boxed().map(c->train_data.subList(c*mini_batch_size, (c+1)*mini_batch_size))
                    .collect(Collectors.toList());
            System.out.println(mini_batches.size());
            long beginPoint=System.currentTimeMillis();
            mini_batches.parallelStream().forEach(mini_batch->{
                    update_mini_batch(mini_batch,eta);
            });
            System.out.println("\n"+(System.currentTimeMillis()-beginPoint));
            if(len_test>0){
                System.out.println("Epoch {"+i+"}: {"+evaluate(test_data[0])+"} // {"+len_test+"}");
            }
            else{
                System.out.println("Epoch {"+i+"}: completed");
            }
        }
    }
    
    public void update_mini_batch(List<INDArray[]> mini_batch, double eta){
        INDArray[][] nablas=mini_batch.parallelStream().map(mb->backprop(mb[0],mb[1]))
                .reduce((acc,x)->{
            for(int i=0;i<num_layers-1;i++){
                    acc[0][i].add(x[0][i]);
                    acc[1][i].add(x[1][i]);
            }
            return acc;
        }).get(); 
        for(int i=0;i<shape.length-1;i++){
            nablas[1][i].mul(eta/mini_batch.size());
            nablas[0][i].mul(eta/mini_batch.size());
            //System.out.println(weights.size());
            //System.out.println(Arrays.toString(weights.get(i).shape()));
            weights.set(i, weights.get(i).sub(nablas[1][i]));
            biases.set(i,biases.get(i).sub(nablas[0][i]));
        }
        System.out.print("-");
    }
    
    public INDArray[][] backprop(INDArray x,INDArray y){
        INDArray[] nabla_b=new INDArray[num_layers-1];
        INDArray [] nabla_w=new INDArray[num_layers-1];
        INDArray activation=x;
        List<INDArray> activations=new LinkedList<>();
        activations.add(x);
        List<INDArray> zs=new LinkedList<>();
        for(int i=0;i<weights.size();i++){
            zs.add(weights.get(i).mmul(activation).add(biases.get(i)));
            activation=Transforms.sigmoid(zs.get(i),true);
            activations.add(activation);
            //System.out.println(Arrays.toString(weights.get(i).shape()));
        }
        INDArray delta=cost_derivative(activations.get(activations.size()-1),y)
                .mul(Transforms.sigmoidDerivative(zs.get(zs.size()-1),true));
        nabla_b[nabla_b.length-1]=delta.dup();
        nabla_w[nabla_w.length-1]=delta.reshape(new int[]{(int)delta.shape()[0],1})
                .mmul(activations.get(activations.size()-2).reshape(
                        new int[]{1,(int)activations.get(activations.size()-2).shape()[0]}));
        for(int i=2;i<num_layers;i++){
            int j=i;
            INDArray z=zs.get(zs.size()-i);
            INDArray sp=Transforms.sigmoidDerivative(z,true);
            delta=weights.get(weights.size()-i+1).transpose().mmul(delta).mul(sp);
            nabla_b[nabla_b.length-i]=delta.dup();
            /*nabla_w[nabla_w.length-i]=new Matrix(nabla_b[nabla_b.length-i]
                .getElements().stream().map(e->
                        activations.get(activations.size()-j-1).getElements().stream()
                        .map(del->del.doubleValue()*e.doubleValue()))
                .flatMap(Function.identity()).collect(Collectors.toList()),
        weights.get(weights.size()-i).getShape(),Double.class);*/
            nabla_w[nabla_w.length-i]=delta.reshape(new int[]{(int)delta.shape()[0],1})
                .mmul(activations.get(activations.size()-j-1).reshape(
                        new int[]{1,(int)activations.get(activations.size()-j-1).shape()[0]}));
        }
        return new INDArray[][]{nabla_b,nabla_w};
    }
    
    public long evaluate(List<INDArray[]> test_data){
        return test_data.parallelStream().filter(c->
                feedforward(c[0]).argMax(0).data().asInt()[0]==c[1].data().asInt()[0])
                .count();
    }
    
    public INDArray cost_derivative(INDArray activations_output, INDArray expected_output){
        return activations_output.sub(expected_output);
    }
    
    public double sigmoid(double z){
        return 1/(1+Math.exp(-z));
    }
    
    public double sigmoid_prime(double z){
        return sigmoid(z)*(1-sigmoid(z));
    }

    public List<INDArray> getWeights() {
        return weights;
    }

    public List<INDArray> getBiases() {
        return biases;
    }
    

    public void setWeights(List<INDArray> weights) {
        this.weights = weights;
    }

    public void setBiases(List<INDArray> biases) {
        this.biases = biases;
    }
}
