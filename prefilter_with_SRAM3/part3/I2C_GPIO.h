#include <Arduino.h>

#define I2C_SCL                      8
#define I2C_SDA                      9

// ----------------------------------------------------------------------------
// Tool API
// ----------------------------------------------------------------------------
void SDA_Low();
void SDA_Release(); 

void SCL_Low();
void SCL_Release();

bool SDA_Read();
void delay_t();


// ----------------------------------------------------------------------------
// Software I2C Primitive Operations
// ----------------------------------------------------------------------------
void I2C_start();

void I2C_repeated_start();

void I2C_stop();

bool I2C_read_bit();

bool I2C_write_byte(uint8_t data);

uint8_t I2C_read_byte(bool ack);
