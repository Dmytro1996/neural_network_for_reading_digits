/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuralnetwork.cost;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author dmytr
 */
public interface Cost {
    
    double fun(INDArray actual_output, INDArray expected_output);
    INDArray delta(INDArray actual_output, INDArray expected_output, INDArray z);
}
