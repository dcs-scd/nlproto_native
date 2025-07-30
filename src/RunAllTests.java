package nl.proto;

import java.nio.file.Paths;

public final class RunAllTests {
    public static void main(String[] args) throws Exception {
        String model = (args.length > 0 ? args[0] : "models/sample.nlogo");
        System.out.println("=== Running NetLogo-native-pipe smoke test ===");
        
        try {
            // Initialize the runtime with model path
            NativeRuntime.init(model);
            
            // Run a simple test
            double result = NativeRuntime.run(50, 1.0, 10.0, 42);
            System.out.println("Test result: " + result);
            System.out.println("Test finished successfully!");
            
        } catch (Exception e) {
            System.err.println("Test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
}