//
//  ViewController.swift
//  MetalMatrixMult
//
//  Created by Anton Akusok on 05/07/2018.
//  Copyright © 2018 Anton Akusok. All rights reserved.
//

import UIKit
import MetalPerformanceShaders
import Accelerate

protocol Logger {
    func log(_ message: String)
}

class ViewController: UIViewController, Logger {
    
    @IBOutlet weak var textField: UITextView!
    @IBOutlet weak var inputTextField: UITextField!
    
    @IBAction func coolButtonPressed(_ sender: Any) {
        let textN = inputTextField.text ?? ""
        let N = Int(textN) ?? 100
        
        textField.text = nil
        self.log("Cool button pressed with number \(N)")
        
        let device = MTLCreateSystemDefaultDevice()!
        guard MPSSupportsMTLDevice(device) else { fatalError("Error: This device has no Metal Performance Shaders") }
        self.log("Running on \(device.name)")
        self.log("Has iOS_GPUFamily4_v2: \(device.supportsFeatureSet(.iOS_GPUFamily4_v2))")
        self.log("Has iOS_GPUFamily4_v1: \(device.supportsFeatureSet(.iOS_GPUFamily4_v1))")
        self.log("Has iOS_GPUFamily3_v4: \(device.supportsFeatureSet(.iOS_GPUFamily3_v4))")
        self.log("Has iOS_GPUFamily3_v3: \(device.supportsFeatureSet(.iOS_GPUFamily3_v3))")
        self.log("Has iOS_GPUFamily3_v2: \(device.supportsFeatureSet(.iOS_GPUFamily3_v2))")
        self.log("Has iOS_GPUFamily3_v1: \(device.supportsFeatureSet(.iOS_GPUFamily3_v1))")
        let maxBS = device.maxBufferLength
        self.log("Max buffer size is: \(maxBS / 1024 / 1024) MB.")
        self.log("Max matrix size is: \(Int(pow(Double(maxBS / 4 ), 0.5)))")
        self.log("")
        
        let commandQueue = device.makeCommandQueue()!
        let commandBuffer = commandQueue.makeCommandBuffer()!
        
        let a = UnsafeMutablePointer<Float>.allocate(capacity: N * N)
        let arrayA = UnsafeMutableBufferPointer(start: a, count: N * N)
        arrayA.update(repeating: Float(1.0))
        self.log("Made array size \(N)*\(N) filled with \(arrayA[0])")

        let arrayA_fp16 = float32to16(a, count: N*N)
        
        let rowBytes = N * MemoryLayout<UInt16>.stride
        let bufferA = device.makeBuffer(bytes: arrayA_fp16.baseAddress!, length: N * rowBytes, options: [])!
        let bufferC = device.makeBuffer(length: N * rowBytes, options: [])!
        
        let descrM = MPSMatrixDescriptor(rows: N, columns: N, rowBytes: rowBytes, dataType: .float16)
        let matrixA = MPSMatrix(buffer: bufferA, descriptor: descrM)
        let matrixC = MPSMatrix(buffer: bufferC, descriptor: descrM)
        
        self.log("Encoding computations...")
        var startTime = CACurrentMediaTime()
            let matMul = MPSMatrixMultiplication(device: device, resultRows: N, resultColumns: N, interiorColumns: N)
            matMul.encode(commandBuffer: commandBuffer, leftMatrix: matrixA, rightMatrix: matrixA, resultMatrix: matrixC)
        var elapsed = CACurrentMediaTime() - startTime
        self.log("Encoding took \(elapsed) seconds")
        
        startTime = CACurrentMediaTime()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        elapsed = CACurrentMediaTime() - startTime
        self.log("Computations done!")
        self.log("Computing tool \(elapsed) seconds")
        
        let ops = 2.0 * pow(Double(N), 3)
        let gflops = ops / elapsed / 1E9
        self.log("Computed at \(Int(gflops)) GFlops")
        
        let resultsPointer = bufferC.contents().bindMemory(to: UInt16.self, capacity: N * N)
        let results = float16to32(resultsPointer, count: N * N)
        self.log("Resulting values: [\(results[0])...\(results[results.count - 1])]")
        
        arrayA_fp16.deallocate()
        arrayA.deallocate()
    }
    
    @IBAction func detailDebugPressed(_ sender: Any) {
        let textN = inputTextField.text ?? ""
        let largeN = Int(textN) ?? 100
        
        textField.text = nil
        self.log("Detailed debig requested with max number \(largeN)")
        
        let device = MTLCreateSystemDefaultDevice()!
        guard MPSSupportsMTLDevice(device) else { fatalError("Error: This device has no Metal Performance Shaders") }
        self.log("Running on \(device.name) \n")
        
        let commandQueue = device.makeCommandQueue()!

        let a = UnsafeMutablePointer<Float>.allocate(capacity: largeN * largeN)
        let arrayA = UnsafeMutableBufferPointer(start: a, count: largeN * largeN)
        arrayA.update(repeating: Float(1.0))
        self.log("Made array size \(largeN)*\(largeN) filled with \(arrayA[0])")
        let arrayA_fp16 = float32to16(a, count: largeN * largeN)

        let rowBytes = largeN * MemoryLayout<UInt16>.stride
        let bufferA = device.makeBuffer(bytes: arrayA_fp16.baseAddress!, length: largeN * rowBytes, options: [])!
        let bufferC = device.makeBuffer(length: largeN * rowBytes, options: [])!

        var startTime = CACurrentMediaTime()
        var elapsed = CACurrentMediaTime()
        var gflops = [Double]()
        var Ns = [Int]()
        var N = 0

        for N1 in 1...largeN/16 {
            N = N1 * 16
            let rowBytes = N * MemoryLayout<UInt16>.stride
            let descrM = MPSMatrixDescriptor(rows: N, columns: N, rowBytes: rowBytes, dataType: .float16)
            
            let matrixA = MPSMatrix(buffer: bufferA, descriptor: descrM)
            let matrixC = MPSMatrix(buffer: bufferC, descriptor: descrM)
            let matMul = MPSMatrixMultiplication(device: device, resultRows: N, resultColumns: N, interiorColumns: N)
            
            let commandBuffer = commandQueue.makeCommandBuffer()!
            matMul.encode(commandBuffer: commandBuffer, leftMatrix: matrixA, rightMatrix: matrixA, resultMatrix: matrixC)
            
            startTime = CACurrentMediaTime()
                commandBuffer.commit()
                commandBuffer.waitUntilCompleted()
            elapsed = CACurrentMediaTime() - startTime
            
            let gf = 2.0 * pow(Double(N), 3) / elapsed / 1E9
            gflops.append(gf)
            Ns.append(N)
            
            print("N = \(N) at \(Int(gf)) GFlops")
        }
        
        print(Ns)
        print(gflops.map { Int($0) })
        arrayA_fp16.deallocate()
        arrayA.deallocate()
    }
    
    @IBAction func blasButtonPressed(_ sender: Any) {
        let textN = inputTextField.text ?? ""
        let N = Int(textN) ?? 100
        
        textField.text = nil
        self.log("Detailed BLAS requested with max number \(N)")
        
        let a = UnsafeMutablePointer<Float>.allocate(capacity: N * N)
        let arrayA = UnsafeMutableBufferPointer(start: a, count: N * N)
        arrayA.update(repeating: Float(1.0))
        self.log("Made array size \(N)*\(N) filled with \(arrayA[0])")

        let c = UnsafeMutablePointer<Float>.allocate(capacity: N * N)
        let arrayC = UnsafeMutableBufferPointer(start: c, count: N * N)
        arrayC.update(repeating: Float(0.0))

        var startTime = CACurrentMediaTime()
        var elapsed = CACurrentMediaTime()
        var gflops = [Double]()
        var Ns = [Int]()
        var N1 = 0
        
        for N_8 in 1...N/8 {
            N1 = N_8 * 8
            
            startTime = CACurrentMediaTime()
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(N1), Int32(N1), Int32(N1), 1,
                        arrayA.baseAddress!, Int32(N1), arrayA.baseAddress!, Int32(N1), 0, arrayC.baseAddress!, Int32(N1))
            elapsed = CACurrentMediaTime() - startTime
            
            let gf = 2.0 * pow(Double(N1), 3) / elapsed / 1E9
            gflops.append(gf)
            Ns.append(N1)
            
            print("N = \(N1) at \(Int(gf)) GFlops")
        }
        
        print(Ns)
        print(gflops.map { Int($0) })
    }

    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        textField.text = nil
        inputTextField.text = "32"
        //coolButtonPressed(1)
    }

    func log(_ message: String) {
        let currentText = textField.text ?? ""
        textField.text = currentText + message + "\n"
    }
    
}

