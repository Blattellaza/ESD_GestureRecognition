#include "I2C_GPIO.h"

// ----------------------------------------------------------------------------
// GPIO Tool APIs for Software I2C
// ------------

void SDA_Low()
{
  pinMode(I2C_SDA, OUTPUT);
  digitalWrite(I2C_SDA, LOW);
}

// Release SDA (High-Z). SDA will be pulled HIGH by pull-up resistor.
void SDA_Release() 
{
  pinMode(I2C_SDA, INPUT);
}

// Drive SCL LOW
void SCL_Low()
{
  pinMode(I2C_SCL, OUTPUT);
  digitalWrite(I2C_SCL, LOW);
}

// Release SCL (High-Z). SCL will be pulled HIGH by pull-up resistor.
void SCL_Release()
{
  pinMode(I2C_SCL, INPUT);
}

// Read SDA logic level
bool SDA_Read()
{
  return (digitalRead(I2C_SDA) != LOW);
}

void delay_t()
{
  delayMicroseconds(1);
}


// ----------------------------------------------------------------------------
// Software I2C Primitive Operations
// ----------------------------------------------------------------------------

// Generate I2C START condition
void I2C_start()
{
  SDA_Release();
  delay_t();
  SCL_Release();
  delay_t();
  SDA_Low();
  delay_t();
  SCL_Low();
  delay_t();
}


void I2C_repeated_start()
{
  SDA_Release();
  delay_t();
  SCL_Release();
  delay_t();
  SDA_Low();
  delay_t();
  SCL_Low();
  delay_t();
}

// Generate I2C STOP condition
void I2C_stop()
{
  SDA_Low();
  delay_t();
  SCL_Release();
  delay_t();
  SDA_Release();    // SDA: L -> H while SCL = H
  delay_t();
}

// Read one bit from the I2C bus: release SDA, pulse SCL high, sample SDA, pull SCL low
bool I2C_read_bit()
{
  SDA_Release();     // slave drives SDA
  delay_t();

  SCL_Release();    // clock high
  delay_t();

  bool bit = SDA_Read();

  SCL_Low();        // clock low
  delay_t();

  return bit;
}


// Write one byte (MSB-first) and read ACK/NACK from slave
// Return true if ACK received (ACK=0), false if NACK (ACK=1)
bool I2C_write_byte(uint8_t data)
{
  // Enter data phase
  bool NAck;
  //  Transmit Slave addr to I2C bus
  for(int i = 0; i < 8; i++)
  {
    if(data & 0x80) SDA_Release();
    else             SDA_Low();

    SCL_Release();
    delay_t();
    SCL_Low();
    delay_t();
    data = data << 1;
  }
  NAck = I2C_read_bit();

  return !NAck;
}


// Read one byte (MSB-first) and then drive ACK/NACK on 9th clock
// ack=true  -> master sends ACK (LOW) to request more bytes
// ack=false -> master sends NACK (HIGH) to end reading
uint8_t I2C_read_byte(bool ack)
{
  uint8_t data = 0;

  for (int i = 7; i >= 0; i--)
  {
    if (I2C_read_bit())
      data |= (1 << i);
  }

  if (ack)
    SDA_Low();      // ACK = 0
  else
    SDA_Release();  // NACK = 1

  delay_t();
  SCL_Release();
  delay_t();
  SCL_Low();
  delay_t();
  SDA_Release();

  return data;
}