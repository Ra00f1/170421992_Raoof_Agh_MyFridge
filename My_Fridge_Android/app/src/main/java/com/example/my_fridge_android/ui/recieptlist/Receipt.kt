package com.example.my_fridge_android.ui.recieptlist

data class Receipt(
    val id: String,
    val storeName: String,
    val date: String,
    val total: String,
    val imageUri: String? = null
)