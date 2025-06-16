package com.example.dbapi;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class DbapiApplication 
{

    public static void main(String[] args) 
	{
		// Türkçe dil desteği için gerekli ayarlar
		System.setProperty("file.encoding", "UTF-8");
        SpringApplication.run(DbapiApplication.class, args);
    }
}

