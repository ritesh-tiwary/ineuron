# ineuron
Assignments - Full Stack Data Science Bootcamp

# Generate a self-signed certificate directly
cert = x509.CertificateBuilder().subject_name(x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, common_name)
    ])).issuer_name(x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, common_name)
    ])).public_key(private_key.public_key()).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.datetime.utcnow()
    ).not_valid_after(
        datetime.datetime.utcnow() + datetime.timedelta(days=365)
    ).sign(private_key, hashes.SHA256(), default_backend())


with open(filename, 'wb') as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    print("Certificate saved to", filename)
