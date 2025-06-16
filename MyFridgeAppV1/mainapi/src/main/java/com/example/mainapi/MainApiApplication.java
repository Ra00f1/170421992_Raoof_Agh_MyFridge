package com.example.mainapi;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class MainApiApplication 
{
	public static void main(String[] args) 
	{
		System.setProperty("file.encoding", "UTF-8");
		SpringApplication.run(MainApiApplication.class, args);
	}
}