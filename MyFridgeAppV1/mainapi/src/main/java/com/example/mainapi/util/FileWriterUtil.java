package com.example.mainapi.util;

import com.example.mainapi.model.Product;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Component;

import java.io.OutputStreamWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

@Component
public class FileWriterUtil 
{
    private static final String FILE_PATH = "products.json";

    public void writeToFile(Product product) 
    {
        try (OutputStreamWriter writer = new OutputStreamWriter(new FileOutputStream(FILE_PATH, true), StandardCharsets.UTF_8)) 
        {
            ObjectMapper mapper = new ObjectMapper();
            String json = mapper.writeValueAsString(product);
            writer.write(json);
            writer.write(System.lineSeparator());
        } 
        catch (IOException e) 
        {
            e.printStackTrace();
        }
    }
}
