/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuranetwork.datastore;

import com.google.gson.Gson;
import com.mycompany.neuralnetwork.matrix.Matrix;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 *
 * @author dmytr
 */
public class DataStore {
    
    private List<String> data;
    private static Gson gson=new Gson();

    public DataStore(List<String> data) {
        this.data = data;
    }
    
    public Matrix[] getPair(int index){
        Matrix[] result=gson.fromJson(data.get(index), Matrix[].class);
        result[0].setElType(Double.class);
        result[1].setElType(Double.class);
        return result;
    }
    
    public List<Matrix[]> subList(int beginIndex, int endIndex){
        return data.subList(beginIndex, endIndex).stream().map(s->{
            Matrix[] m=gson.fromJson(s, Matrix[].class);
            m[0].setElType(Double.class);
            m[1].setElType(Double.class);
            return m;
        }).collect(Collectors.toList());
    }
    
    public void shuffle(){
        Collections.shuffle(data);
    }
    
    public Stream<Matrix[]> stream(){
        return data.stream().map(s->{
            Matrix[] m=gson.fromJson(s, Matrix[].class);
            m[0].setElType(Double.class);
            m[1].setElType(Double.class);
            return m;});
    }

    public List<String> getData() {
        return data;
    }
    
    public int size(){
        return data.size();
    }

    public void setData(List<String> data) {
        this.data = data;
    }
    
    
}
