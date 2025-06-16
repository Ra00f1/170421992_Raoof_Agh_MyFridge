package com.example.OCR_API;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class OCR_APIApplication 
{
    public static void main(String[] args) 
	{
        System.setProperty("file.encoding", "UTF-8");
        SpringApplication.run(OCR_APIApplication.class, args);
    }
}

