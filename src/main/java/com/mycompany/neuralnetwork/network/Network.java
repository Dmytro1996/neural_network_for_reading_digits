/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuralnetwork.network;

import com.mycompany.neuralnetwork.exceptions.MatrixException;
import com.mycompany.neuralnetwork.matrix.Matrix;
import com.mycompany.neuralnetwork.matrix.Matrices;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 *
 * @author dmytr
 */
public class Network {
    
    private int num_layers;
    private int[] shape;
    private List<Matrix> weights;
    private List<Matrix> biases;
    
    public Network(int[] shape) throws MatrixException{
        this.shape=shape;
        this.num_layers=shape.length;
        weights=new LinkedList<>();
        biases=new LinkedList<>();
        for(int i=0;i<shape.length;i++){
            if(i<shape.length-1)weights.add(new Matrix(new int[]{shape[i+1],shape[i]},Double.class));
            if(i>0)biases.add(new Matrix(new int[]{shape[i]},Double.class));//check type for biases
        }
    }
    
    public Matrix feedforward(Matrix activations) throws MatrixException, Exception{
        for(int i=0;i<weights.size();i++){
            activations=Matrices.applyFunc(Matrices.add(Matrices.matrixMult(
                    weights.get(i), activations),biases.get(i)),this::sigmoid);
        }
        return activations;
    }
    
    public void SGD(List<Matrix[]> train_data, int epochs, int mini_batch_size, double eta,List<Matrix[]>...test_data){
        int len_test=(test_data.length!=0)?test_data[0].size():0;
        int len_train=train_data.size();
        for(int i=0;i<epochs;i++){
            Collections.shuffle(train_data);
            List<List<Matrix[]>> mini_batches=IntStream.range(0,(len_train/mini_batch_size))
                    .boxed().map(c->train_data.subList(c*mini_batch_size, (c+1)*mini_batch_size))
                    .collect(Collectors.toList());
            System.out.println(mini_batches.size());
            long beginPoint=System.currentTimeMillis();
            mini_batches.parallelStream().forEach(mini_batch->{
                try {                
                    update_mini_batch(mini_batch,eta);
                } catch (Exception ex) {
                    Logger.getLogger(Network.class.getName()).log(Level.SEVERE, null, ex);
                }
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
    
    public void update_mini_batch(List<Matrix[]> mini_batch, double eta) throws Exception{
        Matrix[][] nablas=mini_batch.parallelStream().map(mb->{
            Matrix[][] result=new Matrix[2][num_layers-1];
            try {
                result=backprop(mb[0],mb[1]);
            } catch (Exception ex) {
                Logger.getLogger(Network.class.getName()).log(Level.SEVERE, null, ex);
            }
            return result;
        }).reduce((acc,x)->{
            for(int i=0;i<num_layers-1;i++){
                try {
                    acc[0][i].add(x[0][i]);
                    acc[1][i].add(x[1][i]);
                } catch (MatrixException ex) {
                    Logger.getLogger(Network.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
            return acc;
        }).get();
        for(int i=0;i<shape.length-1;i++){
            nablas[1][i].multiply(eta/mini_batch.size());
            nablas[0][i].multiply(eta/mini_batch.size());
            weights.get(i).subtract(nablas[1][i]);
            biases.get(i).subtract(nablas[0][i]);
        }
        System.out.print("-");
    }
    
    public Matrix[][] backprop(Matrix x,Matrix y) throws MatrixException, Exception{
        long beginPoint=System.currentTimeMillis();
        Matrix[] nabla_b=new Matrix[shape.length-1];
        Matrix[] nabla_w=new Matrix[shape.length-1];
        Matrix activation=x;
        List<Matrix> activations=new LinkedList<>();
        activations.add(x);
        List<Matrix> zs=new LinkedList<>();
        long zsTime=0;
        long actTime=0;
        for(int i=0;i<weights.size();i++){
            long beginZs=System.currentTimeMillis();
            zs.add(Matrices.add(Matrices.matrixMult(
                    weights.get(i),activation),biases.get(i)));
            activation=Matrices.applyFunc(zs.get(i),this::sigmoid);
            activations.add(activation);
        }
        Matrix delta=Matrices.multiply(cost_derivative(activations.get(activations.size()-1),y),
                Matrices.applyFunc(zs.get(zs.size()-1),this::sigmoid_prime));
        nabla_b[nabla_b.length-1]=Matrices.copyOf(delta);
        nabla_w[nabla_w.length-1]=new Matrix(nabla_b[nabla_b.length-1]
                .getElements().stream().map(e->
                        activations.get(activations.size()-2).getElements().stream()
                        .map(del->del.doubleValue()*e.doubleValue()))
                .flatMap(Function.identity()).collect(Collectors.toList()),
        weights.get(weights.size()-1).getShape(),Double.class);
        for(int i=2;i<num_layers;i++){
            int j=i;
            Matrix z=zs.get(zs.size()-i);
            Matrix sp=Matrices.applyFunc(z, this::sigmoid_prime);
            delta=Matrices.multiply(Matrices.matrixMult(Matrices.transpose(weights.get(weights.size()-i+1)),delta),sp);
            nabla_b[nabla_b.length-i]=Matrices.copyOf(delta);
            nabla_w[nabla_w.length-i]=new Matrix(nabla_b[nabla_b.length-i]
                .getElements().stream().map(e->
                        activations.get(activations.size()-j-1).getElements().stream()
                        .map(del->del.doubleValue()*e.doubleValue()))
                .flatMap(Function.identity()).collect(Collectors.toList()),
        weights.get(weights.size()-i).getShape(),Double.class);
        }
        return new Matrix[][]{nabla_b,nabla_w};
    }
    
    public long evaluate(List<Matrix[]> test_data){
        return test_data.parallelStream().filter(c->{
            try {
                return feedforward(c[0]).argmax()==c[1].getElements().get(0).intValue();
            } catch (Exception ex) {
                Logger.getLogger(Network.class.getName()).log(Level.SEVERE, null, ex);
            }
            return false;
        }).count();
    }
    
    public Matrix cost_derivative(Matrix activations_output, Matrix expected_output) throws MatrixException{
        return Matrices.subtract(activations_output, expected_output);
    }
    
    public double sigmoid(double z){
        return 1/(1+Math.exp(-z));
    }
    
    public double sigmoid_prime(double z){
        return sigmoid(z)*(1-sigmoid(z));
    }

    public List<Matrix> getWeights() {
        return weights;
    }

    public List<Matrix> getBiases() {
        return biases;
    }
    

    public void setWeights(List<Matrix> weights) {
        this.weights = weights;
    }

    public void setBiases(List<Matrix> biases) {
        this.biases = biases;
    }
    
}


