/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuralnetwork.network;

import com.google.gson.Gson;
import com.mycompany.neuralnetwork.cost.Cost;
import com.mycompany.neuralnetwork.cost.CrossEntropyCost;
import com.mycompany.neuralnetwork.neuron.Neuron;
import java.io.BufferedReader;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.ops.NDMath;
import org.nd4j.linalg.ops.transforms.Transforms;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

/**
 *
 * @author dmytr
 */
public class NetworkEnhanced {
    
    private int num_layers;
    private int[] shape;
    private List<INDArray> weights;
    private List<INDArray> biases;
    private Cost cost;
    private Neuron neuron;
    
    public NetworkEnhanced(int[] shape, Cost cost, Neuron neuron){
        this.shape=shape;
        this.num_layers=shape.length;
        weights=new LinkedList<>();
        biases=new LinkedList<>();
        this.cost=cost;
        this.neuron=neuron;
        default_weight_initializer();
    }
    
    public INDArray feedforward(INDArray activation){
        for(int i=0;i<weights.size();i++){
            activation=neuron.fun(weights.get(i).mmul(activation).add(biases.get(i)));
        }
        return activation;
    }
    
    public long countAccuracy(List<INDArray[]> eval_data, boolean convert){
        if(convert){
        return eval_data.parallelStream().filter(c->
                feedforward(c[0]).argMax(0).data().asInt()[0]==c[1].argMax(0).data().asInt()[0])
                .count();
        }
        return eval_data.parallelStream().filter(c->
                feedforward(c[0]).argMax(0).data().asInt()[0]==c[1].data().asInt()[0])
                .count();
    }
    
    public double countTotalCost(List<INDArray[]> data, double lambda){
        double weightsSquare=weights.parallelStream().map(w->new NDMath().square(w.dup()).sum().data().asDouble()[0])
                .reduce((acc,x)->acc+x).orElse(0d);
        return data.parallelStream().map(arr->{
            if(arr[1].data().length()==1) return cost.fun(feedforward(arr[0]), vectorized_result(arr[1]));
            return cost.fun(feedforward(arr[0]), arr[1]);
                }).reduce((acc,x)->acc+x).orElse(0d)+0.5*(lambda/data.size())*weightsSquare;
    }
    
    public INDArray vectorized_result(INDArray expected_output){
        return Nd4j.zeros(shape[num_layers-1]).putScalar(new int[]{(int)expected_output.data().asDouble()[0]},1d);
    }
    
    public void large_weight_initializer(){
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
    
    public void default_weight_initializer(){
        Random rand=new Random();
        for(int i=0;i<num_layers;i++){
            int j=i;
            if(i<num_layers-1)weights.add(Nd4j.create(DoubleStream.generate(()->rand.nextGaussian()/Math.sqrt(shape[j]))
                    .limit(shape[i+1]*shape[i]).toArray(),
                    new long[]{shape[i+1],shape[i]},DataType.DOUBLE));
            if(i>0)
                biases.add(Nd4j.create(DoubleStream.generate(()->rand.nextGaussian())
                        .limit(shape[i]).toArray(),new long[]{shape[i]},DataType.DOUBLE));
        }
    }
    
    public void save(String fileName) throws IOException{
        BufferedWriter br=new BufferedWriter(new FileWriter(fileName, true));
        Gson gson=new Gson();
        br.write(gson.toJson(this, NetworkEnhanced.class));
        br.close();
    }
    
    public NetworkEnhanced load(String fileName) throws IOException{
        FileReader fr=new FileReader(fileName);
        BufferedReader br=new BufferedReader(fr);
        StringBuilder text=new StringBuilder();
        String s="";
        while((s=br.readLine())!=null){
            text.append(s);
        }
        NetworkEnhanced loadedNet=new Gson().fromJson(text.toString(), NetworkEnhanced.class);
        this.weights=loadedNet.getWeights();
        this.biases=loadedNet.getBiases();
        return loadedNet;
    }

    public List<INDArray> getWeights() {
        return weights;
    }

    public List<INDArray> getBiases() {
        return biases;
    }

    public int getNum_layers() {
        return num_layers;
    }

    public int[] getShape() {
        return shape;
    }

    public Cost getCost() {
        return cost;
    }

    public Neuron getNeuron() {
        return neuron;
    }
    

    public void setWeights(List<INDArray> weights) {
        this.weights = weights;
    }

    public void setBiases(List<INDArray> biases) {
        this.biases = biases;
    }
}
